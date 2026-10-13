"""``python -m anodsim`` — GUI 실행."""


def main() -> int:
    from .gui.app import run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
