import argparse

from core.governance_data import GovernanceDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Build governance embeddings cache from the Excel workbook.")
    parser.add_argument("--excel", default=None, help="Path to nodes_edges_governance.xlsx (defaults to constants.EXCEL_PATH).")
    args = parser.parse_args()

    GovernanceDataLoader(excel_path=args.excel)
    print("Built embeddings cache successfully.")


if __name__ == "__main__":
    main()
