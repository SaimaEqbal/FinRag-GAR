# import pandas as pd

# df = pd.read_csv(
#     "financial_data/FY2026_27_corpus/processed_data/combined_annual_corpus.csv"
# )

# keywords = [
#     "revenue",
#     "expense",
#     "profit",
#     "asset",
#     "liabilit"
# ]

# for company in df["company"].unique():
#     print(f"\n{'='*60}")
#     print(company)
#     print("="*60)

#     company_df = df[df["company"] == company]

#     for keyword in keywords:
#         matches = company_df[
#             company_df["metric"].str.contains(
#                 keyword, case=False, na=False
#             )
#         ]

#         print(f"\n--- {keyword.upper()} ---")
#         print(
#             matches[
#                 ["metric", "value", "unit", "segment", "text"]
#             ].head(10).to_string(index=False)
#         )

import pandas as pd

df = pd.read_csv("financial_data/FY2026_27_corpus/processed_data/combined_annual_corpus.csv")

important_metrics = [
    "RevenueFromOperations",
    "ProfitLossForPeriod",
    "ProfitOrLossAttributableToOwnersOfParent",
    "Assets",
    "Liabilities",
    "Equity",
    "EquityAndLiabilities"
]

for company in df["company"].unique():
    print(f"\n--- {company} ---")

    company_df = df[df["company"] == company]

    for metric in important_metrics:
        rows = company_df[
            company_df["metric"].str.lower() == metric.lower()
        ]

        if not rows.empty:
            print(rows[["metric", "value", "unit", "text"]]
                  .to_string(index=False))