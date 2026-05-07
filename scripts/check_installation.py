#!/usr/bin/env python3
"""
Verify CXR-ContraBench installation and dependencies.
"""
import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("\nChecking imports...")
    imports = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
    }

    all_ok = True
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False

    return all_ok


def check_package():
    """Check if cxr_contrabench package can be imported."""
    print("\nChecking cxr_contrabench package...")
    try:
        import cxr_contrabench
        print(f"  ✓ cxr_contrabench {cxr_contrabench.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False

    # Check submodules
    submodules = ['common', 'datasets', 'metrics', 'verifier']
    all_ok = True
    for submodule in submodules:
        try:
            __import__(f'cxr_contrabench.{submodule}')
            print(f"  ✓ cxr_contrabench.{submodule}")
        except ImportError as e:
            print(f"  ✗ cxr_contrabench.{submodule}: {e}")
            all_ok = False

    return all_ok


def check_examples():
    """Check if example files exist."""
    print("\nChecking example files...")
    examples_dir = Path(__file__).parent.parent / "examples"
    files = [
        "minimal_example.py",
        "__init__.py",
    ]

    all_ok = True
    for filename in files:
        filepath = examples_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
            all_ok = False

    return all_ok


def check_structure():
    """Check directory structure."""
    print("\nChecking directory structure...")
    root = Path(__file__).parent.parent
    dirs = [
        "cxr_contrabench",
        "scripts",
        "examples",
        "docs",
    ]

    all_ok = True
    for dirname in dirs:
        dirpath = root / dirname
        if dirpath.exists() and dirpath.is_dir():
            print(f"  ✓ {dirname}/")
        else:
            print(f"  ✗ {dirname}/ not found")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("CXR-ContraBench Installation Check")
    print("=" * 60)

    results = {}
    results['structure'] = check_structure()
    results['package'] = check_package()
    results['imports'] = check_imports()
    results['examples'] = check_examples()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all(results.values()):
        print("\n✓ All checks passed! Installation is OK.")
        print("\nYou can now run:")
        print("  python examples/minimal_example.py")
        print("  python scripts/eval_vqa.py --help")
        print("  python scripts/apply_verifier.py --help")
        return 0
    else:
        print("\n✗ Some checks failed. Please:")
        print("  1. Verify installation with: pip install -e .")
        print("  2. Install dependencies with: pip install -r requirements.txt")
        print("  3. Run this check again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
