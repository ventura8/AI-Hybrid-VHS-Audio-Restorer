import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Import radon for direct complexity calculation
try:
    from radon.complexity import cc_visit
except ImportError:
    cc_visit = None


COMPLEXITY_UNAVAILABLE = "unavailable"


def generate_badge(xml_file, output_path="assets/coverage.svg"):
    xml_path = Path(xml_file)
    if not xml_path.exists():
        print(f"Error: {xml_file} not found")
        sys.exit(1)

    # Ensure output directory exists when a directory component is present.
    output_path_obj = Path(output_path)
    if output_path_obj.parent != Path("."):
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "genbadge.main",
        "coverage",
        "-i",
        str(xml_path),
        "-o",
        str(output_path_obj),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to generate coverage badge with genbadge: {exc.stderr.strip()}")
        sys.exit(1)

    print(f"Generated badge with genbadge: {output_path_obj}")


def _compute_total_coverage_fraction(root):
    """Compute combined line+branch coverage to match pytest-cov TOTAL."""
    try:
        lines_valid = int(root.get("lines-valid", "0"))
        lines_covered = int(root.get("lines-covered", "0"))
        branches_valid = int(root.get("branches-valid", "0"))
        branches_covered = int(root.get("branches-covered", "0"))

        denominator = lines_valid + branches_valid
        if denominator > 0:
            return (lines_covered + branches_covered) / denominator
    except ValueError:
        pass

    # Fallback keeps previous behavior when aggregate counters are unavailable.
    return float(root.get("line-rate", "0"))


def get_complexity(file_path):
    """Calculates average cyclomatic complexity using radon."""
    if cc_visit is None:
        print("radon is not installed. Please install it with: poetry install -v --with dev --no-root")
        return COMPLEXITY_UNAVAILABLE

    try:
        with Path(file_path).open("r", encoding="utf-8") as f:
            code = f.read()

        blocks = cc_visit(code)
        if not blocks:
            return 0

        total_cc = sum(b.complexity for b in blocks)
        return round(total_cc / len(blocks), 2)
    except Exception as e:
        print(f"DEBUG: Complexity calculation Error for {file_path}: {e}")
        return COMPLEXITY_UNAVAILABLE


def _resolve_file_path(filename, source_roots):
    filename_path = Path(filename)
    filename_norm = str(filename_path)
    filename_base = filename_path.name

    candidates = []

    def add_candidate(path):
        if path and path not in candidates:
            candidates.append(path)

    add_candidate(Path(filename_norm))
    add_candidate(Path.cwd() / filename_norm)

    for source in source_roots:
        source_norm = Path(source)
        add_candidate(source_norm / filename_norm)
        add_candidate(source_norm / filename_base)

    add_candidate(Path(filename_base))
    add_candidate(Path.cwd() / filename_base)

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)

    return None


def update_complexity(root):
    """Update complexity for each class using radon."""
    source_roots = []
    for source_el in root.findall("./sources/source"):
        if source_el.text:
            source_roots.append(source_el.text.strip())

    for cls in root.findall(".//class"):
        filename = cls.get("filename")
        if filename and filename.endswith(".py"):
            target_file = _resolve_file_path(filename, source_roots)

            if target_file:
                cc = get_complexity(target_file)
                cls.set("complexity", str(cc))
            else:
                cls.set("complexity", COMPLEXITY_UNAVAILABLE)


def _copy_package_metrics(new_pkg, cls):
    for attr in ["line-rate", "branch-rate", "complexity"]:
        val = cls.get(attr)
        if val is not None:
            new_pkg.set(attr, str(val))
        elif attr == "complexity":
            new_pkg.set(attr, COMPLEXITY_UNAVAILABLE)
        else:
            new_pkg.set(attr, "0.0")


def transform_coverage(xml_file):
    xml_path = Path(xml_file)
    if not xml_path.exists():
        print(f"Error: {xml_file} not found")
        sys.exit(1)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        packages_el = root.find("packages")
        if packages_el is None:
            print("No <packages> element found")
            sys.exit(1)

        # Update complexity for each class using radon
        update_complexity(root)

        # Generate badge from coverage.xml via genbadge.
        generate_badge(xml_path)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)

    # Collect all classes from all existing packages
    all_classes = []
    for pkg in packages_el.findall("package"):
        classes_el = pkg.find("classes")
        if classes_el is not None:
            all_classes.extend(classes_el.findall("class"))

    # Clear existing packages
    packages_el.clear()

    # Create new package per class
    for cls in all_classes:
        filename = cls.get("filename")
        pkg_name = filename if filename else "unknown"

        new_pkg = ET.SubElement(packages_el, "package")
        new_pkg.set("name", pkg_name)
        _copy_package_metrics(new_pkg, cls)

        new_classes = ET.SubElement(new_pkg, "classes")
        new_classes.append(cls)

    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)


def generate_summary(xml_file):
    xml_path = Path(xml_file)
    if not xml_path.exists():
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        total_rate = _compute_total_coverage_fraction(root) * 100
        line_rate = float(root.get("line-rate", "0")) * 100
        branch_rate = float(root.get("branch-rate", "0")) * 100
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("📊 Code Coverage Report")
        print()
        print(f"Total Coverage: {total_rate:.2f}%")
        print(f"Line Coverage: {line_rate:.2f}%")
        print(f"Branch Coverage: {branch_rate:.2f}%")
        print(f"Generated: {generated_at}")
        print()
        print("File | Coverage | Branches | Complexity")
        print("--- | --- | --- | ---")

        packages = root.find("packages")
        if packages is not None:
            for pkg in packages.findall("package"):
                name = pkg.get("name", "unknown")
                l_rate = float(pkg.get("line-rate", "0")) * 100
                b_rate = float(pkg.get("branch-rate", "0")) * 100
                complexity = pkg.get("complexity", COMPLEXITY_UNAVAILABLE)
                print(f"{name} | {l_rate:.1f}% | {b_rate:.1f}% | {complexity}")

    except Exception as e:
        print(f"Error generating summary: {e}")


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "--summary"]
    xml_path = args[0] if args else "coverage.xml"

    if "--summary" in sys.argv[1:]:
        generate_summary(xml_path)
    else:
        transform_coverage(xml_path)
