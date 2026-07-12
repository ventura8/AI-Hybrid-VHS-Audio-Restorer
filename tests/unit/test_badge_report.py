import xml.etree.ElementTree as ET
from unittest.mock import patch

from tests.tooling import badge_report


def test_get_complexity_returns_unavailable_on_failure(tmp_path, capsys):
    source_file = tmp_path / "broken.py"
    source_file.write_text("def broken(:\n", encoding="utf-8")

    result = badge_report.get_complexity(source_file)

    assert result == badge_report.COMPLEXITY_UNAVAILABLE
    captured = capsys.readouterr()
    assert "Complexity calculation Error" in captured.out


def test_transform_coverage_marks_missing_file_complexity_unavailable(tmp_path):
    coverage_file = tmp_path / "coverage.xml"
    coverage_file.write_text(
        """<?xml version=\"1.0\" ?>
<coverage line-rate=\"1\" branch-rate=\"1\" lines-valid=\"1\" lines-covered=\"1\" branches-valid=\"0\" branches-covered=\"0\">
  <sources>
    <source>{source}</source>
  </sources>
  <packages>
    <package name=\"pkg\" line-rate=\"1\" branch-rate=\"1\" complexity=\"0\">
      <classes>
        <class name=\"missing\" filename=\"missing.py\" line-rate=\"1\" branch-rate=\"1\">
          <methods />
          <lines />
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""".format(source=tmp_path.as_posix()),
        encoding="utf-8",
    )

    with patch("tests.tooling.badge_report.generate_badge"):
        badge_report.transform_coverage(str(coverage_file))

    root = ET.parse(coverage_file).getroot()
    package = root.find("packages/package")
    assert package is not None
    assert package.get("complexity") == badge_report.COMPLEXITY_UNAVAILABLE


def test_generate_summary_preserves_unavailable_complexity(tmp_path, capsys):
    coverage_file = tmp_path / "coverage.xml"
    coverage_file.write_text(
        """<?xml version=\"1.0\" ?>
<coverage line-rate=\"1\" branch-rate=\"1\" lines-valid=\"1\" lines-covered=\"1\" branches-valid=\"0\" branches-covered=\"0\">
  <packages>
    <package name=\"sample.py\" line-rate=\"1\" branch-rate=\"1\" complexity=\"unavailable\" />
  </packages>
</coverage>
""",
        encoding="utf-8",
    )

    badge_report.generate_summary(str(coverage_file))

    captured = capsys.readouterr()
    assert "sample.py | 100.0% | 100.0% | unavailable" in captured.out
