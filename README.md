# Cross-Pair Order-Flow Imbalance

Does *relative* order-flow pressure in `ETH/USD` and `BTC/USD` predict the next 5-second `ETH/BTC` return, beyond `ETH/BTC`'s own OFI and current triangular price information?

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a .env file in the repo root with your Massive API key:
```bash
export MASSIVE_API_KEY=...
```
Only `fetch_trades.py` makes calls to the API.

## Usage

Run these three scripts in order from the repo root:

```bash
# fetches 30 days of historical trades for ETH-USD/BTC-USD/ETH-BTC
python fetch_trades.py
# signs trades with the Lee-Ready tick rule, builds the regression features
python process_data.py  
# fits baseline vs extended, runs HAC/bootstrap inference, writes results  
python model.py
```

## Assistance and source disclosure

**Academic papers**
- R. Cont, A. Kukanov, S. Stoikov, "The price impact of order book events,"
  *Journal of Financial Econometrics* 12(1):47–88, 2014.
- R. Cont, M. Cucuringu, C. Zhang, "Cross-impact of order flow imbalance in
  equity markets," arXiv:2112.13213, 2021.
- F. Capponi, R. Cont, "Multi-asset market impact and order flow commonality,"
  working paper, SSRN 3706390, 2020.
- C. M. C. Lee, M. J. Ready, "Inferring trade direction from intraday data,"
  *Journal of Finance* 46(2):733–746, 1991.

**External datasets** — none used for the primary analysis.

**Existing repositories / code consulted** — none directly reused.

**Tutorials / articles** — Massive API documentation:
`massive.com/docs/websocket/quickstart`, `massive.com/docs/rest/quickstart`.

**AI tools**
- Claude Code was used to scaffold the repository, generate the initial version of `fetch_trades.py` and `process_data.py`, and code refactoring. All statistical choices follow the proposal as we reviewed and are responsible for every part.

**Assistance from another person** — none.
