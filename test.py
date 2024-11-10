from PortfolioAnalyser import Engine, optimize_portfolio, PortfolioAnalyser

A = {
    "RELIANCE.NS" : 12500,
    "TCS.NS" : 15000
}

stocks = list(A.keys())
weights = list(A.values())

engine = Engine(
    start_date="2023-04-01",
    portfolio=stocks,
    weights=weights,
    optimizer="HRP"
)

PortfolioAnalyser(engine, report=True)
optimize_portfolio(engine)