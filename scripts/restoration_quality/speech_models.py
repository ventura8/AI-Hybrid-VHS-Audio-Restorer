"""Speech readings that need a model: words unchanged (Whisper), timbre unchanged (WavLM-SV), naturalness (UTMOS).

There is no transcript of a family tape, so ASR consistency is read between the
source's transcript and the output's: a restoration that changed words raises the
character error rate between the two. Speaker cosine is read against the tape's own
intra-speaker floor, because absolute values are model- and channel-specific.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from modules.utils import MODELS_DIR
from scripts.restoration_quality import audio_io

WHISPER_DIR = "whisper-large-v3-turbo"
WAVLM_DIR = "wavlm-base-plus-sv"
UTMOS_DIR = "utmos"
UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
NO_SPEECH_THRESHOLD = 0.6
MAX_NEW_TOKENS = 220
REPEAT_NGRAM = 3
REPEAT_COUNT = 3
FLOOR_PERCENTILE = 10.0
# WavLM-base-plus-sv encoder layer whose mean-pooled hidden states carry the SSL distance.
SSL_LAYER = 6
CEDILLA_TO_COMMA = str.maketrans({"\u015f": "\u0219", "\u0163": "\u021b", "\u015e": "\u0218", "\u0162": "\u021a"})


def _require(models_dir, name, set_name):
    path = Path(models_dir or MODELS_DIR) / name
    if not path.exists():
        raise SystemExit(f"{name} missing under {path.parent}; run scripts/download_quality_models.py --set {set_name}")
    return path


def load_whisper(device, models_dir):
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    path = _require(models_dir, WHISPER_DIR, "speech")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(str(path), torch_dtype=dtype).to(device).eval()
    return model, AutoProcessor.from_pretrained(str(path))


def load_wavlm(device, models_dir):
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    path = _require(models_dir, WAVLM_DIR, "speech")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = WavLMForXVector.from_pretrained(str(path), torch_dtype=dtype).to(device).eval()
    return model, AutoFeatureExtractor.from_pretrained(str(path))


def load_utmos(device, models_dir):
    import torch

    path = _require(models_dir, UTMOS_DIR, "speech")
    torch.hub.set_dir(str(path))
    return torch.hub.load(UTMOS_REPO, "utmos22_strong", trust_repo=True, skip_validation=True).to(device).eval()


def normalise_ro(text):
    """NFC, lowercase, cedilla s/t to comma s/t, punctuation stripped, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text).lower().translate(CEDILLA_TO_COMMA)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def cer_wer(reference, hypothesis):
    """Character and word error rates with jiwer's empty-input cases spelled out."""
    reference, hypothesis = normalise_ro(reference), normalise_ro(hypothesis)
    if not reference and not hypothesis:
        return 0.0, 0.0
    if not reference or not hypothesis:
        return 1.0, 1.0
    import jiwer

    return float(jiwer.cer(reference, hypothesis)), float(jiwer.wer(reference, hypothesis))


def hallucinated(transcript, route):
    """Text the model most likely made up: on a silent window, under a high no-speech probability, or looping."""
    text = normalise_ro(transcript["text"])
    if not text:
        return False
    if route == "silence" or transcript["no_speech_prob"] > NO_SPEECH_THRESHOLD:
        return True
    return _repeats(text.split())


def _repeats(words):
    grams = [" ".join(words[i : i + REPEAT_NGRAM]) for i in range(len(words) - REPEAT_NGRAM + 1)]
    return any(grams.count(gram) >= REPEAT_COUNT for gram in set(grams))


class Whisper:
    """Transcribes 16 kHz windows with the language forced, reading confidence alongside the text."""

    def __init__(self, model, processor, language):
        self.model, self.processor, self.language = model, processor, language
        tokenizer = processor.tokenizer
        self.no_speech_id = tokenizer.convert_tokens_to_ids("<|nospeech|>")
        self.start_id = model.generation_config.decoder_start_token_id

    def transcribe(self, audio16k):
        import torch

        features = self.processor(audio16k, sampling_rate=audio_io.SPEECH_RATE, return_tensors="pt").input_features
        features = features.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(
                features,
                language=self.language,
                task="transcribe",
                num_beams=1,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
            )
            scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            logits = self.model(features, decoder_input_ids=torch.tensor([[self.start_id]], device=self.model.device)).logits[0, -1].float()
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        finite = scores[0][torch.isfinite(scores[0])]
        avg_logprob = float(finite.mean()) if len(finite) else 0.0
        return {"text": text, "avg_logprob": avg_logprob, "no_speech_prob": float(torch.softmax(logits, dim=-1)[self.no_speech_id])}


class TranscriptCache:
    """Transcripts keyed by file and window, so a source is never transcribed twice across variants."""

    def __init__(self, cache_dir, key):
        self.path = Path(cache_dir) / "asr" / f"{key}.json"
        self.records = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get(self, window, produce):
        slot = str(window)
        if slot not in self.records:
            self.records[slot] = produce()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.records, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.records[slot]


def embeddings(model, extractor, audio16k):
    """`(x-vector, SSL embedding)` from one WavLM pass, both unit length.

    The x-vector is the speaker reading. The SSL embedding is the mean over time of the
    encoder's layer `SSL_LAYER` hidden states: its cosine distance between source and output
    is an over-suppression reading that needs no transcript (ArtiFree's detector, 2025).
    """
    import torch

    inputs = extractor(audio16k, sampling_rate=audio_io.SPEECH_RATE, return_tensors="pt", padding=True)
    with torch.inference_mode():
        result = model(
            **{k: v.to(model.device, dtype=model.dtype) if v.dtype.is_floating_point else v.to(model.device) for k, v in inputs.items()},
            output_hidden_states=True,
        )
    hidden = result.hidden_states[min(SSL_LAYER, len(result.hidden_states) - 1)][0].float().mean(dim=0).cpu().numpy()
    return _unit(result.embeddings[0].float().cpu().numpy()), _unit(hidden)


def _unit(vector):
    return vector / (np.linalg.norm(vector) + 1e-9)


def xvector(model, extractor, audio16k):
    return embeddings(model, extractor, audio16k)[0]


def speaker_floor(embeddings):
    """The tape's own intra-speaker floor: p10 of the cosine between source windows two apart (no overlap)."""
    pairs = [
        float(np.dot(embeddings[i], embeddings[i + 2]))
        for i in range(len(embeddings) - 2)
        if embeddings[i] is not None and embeddings[i + 2] is not None
    ]
    return float(np.percentile(pairs, FLOOR_PERCENTILE)) if pairs else None


def utmos_score(predictor, audio16k, device):
    import torch

    with torch.inference_mode():
        wave = torch.from_numpy(np.asarray(audio16k, dtype=np.float32))[None, :].to(device)
        return float(predictor(wave, sr=audio_io.SPEECH_RATE)[0])


def _speech_rows(card):
    return [row for row in card.rows if row.route in ("speech", "mixed")]


def _window(pair, side, row):
    audio = pair.at(side, audio_io.SPEECH_RATE)
    begin, end = int(round(row.start_s * audio_io.SPEECH_RATE)), int(round(row.end_s * audio_io.SPEECH_RATE))
    return audio[begin:end]


def _asr(pair, card, registry):
    model, processor = registry.get("whisper", load_whisper)
    whisper = Whisper(model, processor, getattr(registry, "language", "ro"))
    caches = {
        side: TranscriptCache(pair.cache_dir, f"{key}_{pair.lag}_{whisper.language}")
        for side, key in (("source", pair.source_key), ("output", pair.output_key))
    }
    for row in _speech_rows(card):
        src = caches["source"].get(row.window, lambda: whisper.transcribe(_window(pair, "source", row)))
        out = caches["output"].get(row.window, lambda: whisper.transcribe(_window(pair, "output", row)))
        _asr_row(row, src, out)


def _asr_row(row, src, out):
    row.source["speech.avg_logprob"], row.output["speech.avg_logprob"] = src["avg_logprob"], out["avg_logprob"]
    row.source["speech.no_speech_prob"], row.output["speech.no_speech_prob"] = src["no_speech_prob"], out["no_speech_prob"]
    # The output transcript is judged on the output's own route: text on a window that fell silent is invented.
    row.source["speech.hallucinated"] = 0.0
    row.output["speech.hallucinated"] = float(hallucinated(out, getattr(row, "output_route", "") or row.route))
    if hallucinated(src, row.route):
        return
    cer, wer = cer_wer(src["text"], out["text"])
    row.source["speech.cer"], row.output["speech.cer"] = 0.0, cer
    row.source["speech.wer"], row.output["speech.wer"] = 0.0, wer


def _speaker(pair, card, registry):
    model, extractor = registry.get("wavlm", load_wavlm)
    rows = _speech_rows(card)
    source_embeddings = [embeddings(model, extractor, _window(pair, "source", row)) for row in rows]
    card.speaker_floor = speaker_floor([xvec for xvec, _ssl in source_embeddings])
    for row, (src_xvec, src_ssl) in zip(rows, source_embeddings):
        out_xvec, out_ssl = embeddings(model, extractor, _window(pair, "output", row))
        row.source["speech.speaker_cos"], row.output["speech.speaker_cos"] = 0.0, float(np.dot(src_xvec, out_xvec))
        row.source["speech.ssl_dist"], row.output["speech.ssl_dist"] = 0.0, float(1.0 - np.dot(src_ssl, out_ssl))


def _naturalness(pair, card, registry):
    predictor = registry.get("utmos", load_utmos)
    gain = pair.mos_gain()
    for row in _speech_rows(card):
        row.source["speech.utmos"] = utmos_score(predictor, _window(pair, "source", row) * gain, registry.device)
        row.output["speech.utmos"] = utmos_score(predictor, _window(pair, "output", row) * gain, registry.device)


def score(pair, card, registry):
    """ASR consistency, speaker cosine and UTMOS on the speech and mixed windows, one model at a time."""
    for stage in (_asr, _speaker, _naturalness):
        stage(pair, card, registry)
        registry.release()
