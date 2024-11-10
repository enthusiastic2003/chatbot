# Import the necessary libraries
import numpy as np
import pandas as pd
import csv
import datetime as dt
import quantstats as qs
from IPython.display import display
import matplotlib.pyplot as plt
import copy
import yfinance as yf           # Used to download the stock data
from fpdf import FPDF           # Used to generate the PDF report
import warnings
import logging
from empyrical import (         # Used to calculate the performance metrics
    cagr,
    cum_returns,
    stability_of_timeseries,
    max_drawdown,
    sortino_ratio,
    alpha_beta,
    tail_ratio,
)
from pypfopt import (           # Used to optimize the portfolio
    EfficientFrontier,
    risk_models,
    expected_returns,
    HRPOpt,
    objective_functions,
)
from news_database_interface import interface

warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.legend').disabled = True
TODAY = dt.date.today()
BENCHMARK = ["^NSEI"]           # Default benchamrk used is NIFTY 50
DAYS_IN_YEAR = 365

# Defining our rebalance periods
rebalance_periods = {
    "daily": DAYS_IN_YEAR / 365,
    "weekly": DAYS_IN_YEAR / 52,
    "monthly": DAYS_IN_YEAR / 12,
    "month": DAYS_IN_YEAR / 12,
    "m": DAYS_IN_YEAR / 12,
    "quarterly": DAYS_IN_YEAR / 4,
    "quarter": DAYS_IN_YEAR / 4,
    "q": DAYS_IN_YEAR / 4,
    "6m": DAYS_IN_YEAR / 2,
    "2q": DAYS_IN_YEAR / 2,
    "1y": DAYS_IN_YEAR,
    "year": DAYS_IN_YEAR,
    "y": DAYS_IN_YEAR,
    "2y": DAYS_IN_YEAR * 2,
}

# Defining colors for the allocation pie
CS = [
          "#ff9999",
          "#66b3ff",
          "#99ff99",
          "#ffcc99",
          "#f6c9ff",
          "#a6fff6",
          "#fffeb8",
          "#ffe1d4",
          "#cccdff",
          "#fad6ff",
      ]

aspects = [
    "Earnings",
    "Revenue",
    "Margins",
    "Dividend",
    "EBITDA",
    "Debt",
    "Sentiment"
    ]

class Engine:
    def __init__(
        self,
        start_date,
        portfolio,
        weights=None,
        rebalance=None,
        benchmark=None,
        end_date=TODAY,
        optimizer=None,
        max_vol=0.15,
        diversification=1,
        expected_returns=None,
        risk_model=None,
        min_weights=None,
        max_weights=None,
        risk_manager=None,
        data=pd.DataFrame(),
    ):
        if benchmark is None:
            benchmark = BENCHMARK

        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = portfolio
        self.weights = weights
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.rebalance = rebalance
        self.max_vol = max_vol
        self.diversification = diversification
        self.expected_returns = expected_returns
        if expected_returns is not None:
            assert expected_returns in ["mean_historical_return", "ema_historical_return", "capm_return"], f"Expected return method: {expected_returns} not supported yet! \n Set an appropriate expected returns parameter to your portfolio: mean_historical_return, ema_historical_return or capm_return."
        self.risk_model = risk_model
        if risk_model is not None:
            assert risk_model in ["sample_cov", "semicovariance", "exp_cov", "ledoit_wolf", "ledoit_wolf_constant_variance", "ledoit_wolf_single_factor", "ledoit_wolf_constant_correlation", "oracle_approximating"], f"Risk model: {risk_model} not supported yet! \n Set an appropriate risk model to your portfolio: sample_cov, semicovariance, exp_cov, ledoit_wolf, ledoit_wolf_constant_variance, ledoit_wolf_single_factor, ledoit_wolf_constant_correlation, oracle_approximating."
        self.max_weights = max_weights
        self.min_weights = min_weights
        self.risk_manager = risk_manager
        self.data = data

        optimizers = {
            "EF": efficient_frontier,
            "MEANVAR": mean_var,
            "HRP": hrp,
            "MINVAR": min_var,
        }
        if self.optimizer is None and self.weights is None:
            self.weights = [1.0 / len(self.portfolio)] * len(self.portfolio)
        elif self.optimizer in optimizers.keys():
            if self.optimizer == "MEANVAR":
                self.weights = optimizers.get(self.optimizer)(self, vol_max=max_vol, perf=False)
            else:
                self.weights = optimizers.get(self.optimizer)(self, perf=False)

        if self.rebalance is not None:
            self.rebalance = make_rebalance(
                self.start_date,
                self.end_date,
                self.optimizer,
                self.portfolio,
                self.rebalance,
                self.weights,
                self.max_vol,
                self.diversification,
                self.min_weights,
                self.max_weights,
                self.expected_returns,
                self.risk_model
            )


def get_returns(stocks, wts, start_date, end_date=TODAY):
    if len(stocks) > 1:
        assets = yf.download(stocks, start=start_date, end=end_date, progress=False)["Adj Close"]
        assets = assets.filter(stocks)
        initial_alloc = wts/assets.iloc[0]
        if initial_alloc.isna().any():
            missing_stocks = initial_alloc[initial_alloc.isna()].index
            # Find which stock is not available at the initial state and raise an error
            if len(missing_stocks) == 1:
                raise ValueError(f"{missing_stocks} is not available at initial state!")
            elif len(missing_stocks) > 1:
                raise ValueError(f"These stocks is not available at initial state! {missing_stocks}")
        portfolio_value = (assets * initial_alloc).sum(axis=1)
        returns = portfolio_value.pct_change()[1:]
        return returns
    else:
        df = yf.download(stocks, start=start_date, end=end_date, progress=False)["Adj Close"]
        df = pd.DataFrame(df)
        returns = df.pct_change()[1:]
        return returns


def get_returns_from_data(data, wts, stocks):
    assets = data.filter(stocks)
    initial_alloc = wts/assets.iloc[0]
    if initial_alloc.isna().any():
        missing_stocks = initial_alloc[initial_alloc.isna()].index
        # Find which stock is not available at the initial state and raise an error
        if len(missing_stocks) == 1:
            raise ValueError(f"{missing_stocks} is not available at initial state!")
        elif len(missing_stocks) > 1:
            raise ValueError(f"These stocks is not available at initial state! {missing_stocks}")
    portfolio_value = (assets * initial_alloc).sum(axis=1)
    returns = portfolio_value.pct_change()[1:]
    return returns


def calculate_information_ratio(returns, benchmark_returns, days=252) -> float:
    return_difference = returns - benchmark_returns
    volatility = return_difference.std() * np.sqrt(days)
    information_ratio_result = return_difference.mean() / volatility
    return information_ratio_result


def graph_allocation(my_portfolio):
    fig1, ax1 = plt.subplots()
    ax1.pie(
        my_portfolio.weights,
        labels=my_portfolio.portfolio,
        autopct="%1.1f%%",
        shadow=False,
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Portfolio's allocation")
    #plt.show()


def PortfolioAnalyser(my_portfolio, rf=0.0, sigma_value=1, confidence_value=0.95, report=False, filename="report.pdf"):
    if isinstance(my_portfolio.rebalance, pd.DataFrame):
        # we want to get the dataframe with the dates and weights
        rebalance_schedule = my_portfolio.rebalance

        columns = []

        for date in rebalance_schedule.columns:
            date = date[0:10]
            columns.append(date)
        rebalance_schedule.columns = columns

        # then want to make a list of the dates and start with our first date
        dates = [my_portfolio.start_date]

        # then our rebalancing dates into that list
        dates = dates + rebalance_schedule.columns.to_list()

        datess = []
        for date in dates:
            date = date[0:10]
            datess.append(date)
        dates = datess
        # this will hold returns
        returns = pd.Series()

        # then we want to be able to call the dates like tuples
        for i in range(len(dates) - 1):
            # get our weights
            weights = rebalance_schedule[str(dates[i + 1])]

            # then we want to get the returns
            
            add_returns = get_returns(
                my_portfolio.portfolio,
                weights,
                start_date=dates[i],
                end_date=dates[i + 1],
            )

            # then append those returns
            returns = returns.append(add_returns)
    else:
      if not my_portfolio.data.empty:
              returns = get_returns_from_data(my_portfolio.data, my_portfolio.weights, my_portfolio.portfolio)
      else:
              returns = get_returns(
                  my_portfolio.portfolio,
                  my_portfolio.weights,
                  start_date=my_portfolio.start_date,
                  end_date=my_portfolio.end_date,
              )

    creturns = (returns + 1).cumprod()

    # risk manager
    try:
        if list(my_portfolio.risk_manager.keys())[0] == "Stop Loss":

            values = []
            for r in creturns:
                if r <= 1 + my_portfolio.risk_manager["Stop Loss"]:
                    values.append(r)
                else:
                    pass

            try:
                date = creturns[creturns == values[0]].index[0]
                date = str(date.to_pydatetime())
                my_portfolio.end_date = date[0:10]
                returns = returns[: my_portfolio.end_date]

            except Exception as e:
                pass

        if list(my_portfolio.risk_manager.keys())[0] == "Take Profit":

            values = []
            for r in creturns:
                if r >= 1 + my_portfolio.risk_manager["Take Profit"]:
                    values.append(r)
                else:
                    pass

            try:
                date = creturns[creturns == values[0]].index[0]
                date = str(date.to_pydatetime())
                my_portfolio.end_date = date[0:10]
                returns = returns[: my_portfolio.end_date]

            except Exception as e:
                pass

        if list(my_portfolio.risk_manager.keys())[0] == "Max Drawdown":

            drawdown = qs.stats.to_drawdown_series(returns)

            values = []
            for r in drawdown:
                if r <= my_portfolio.risk_manager["Max Drawdown"]:
                    values.append(r)
                else:
                    pass

            try:
                date = drawdown[drawdown == values[0]].index[0]
                date = str(date.to_pydatetime())
                my_portfolio.end_date = date[0:10]
                returns = returns[: my_portfolio.end_date]

            except Exception as e:
                pass

    except Exception as e:
        pass

    # print("Start date: " + str(my_portfolio.start_date))
    # print("End date: " + str(my_portfolio.end_date))

    benchmark = get_returns(
        my_portfolio.benchmark,
        wts=[1],
        start_date=my_portfolio.start_date,
        end_date=my_portfolio.end_date,
    )
    benchmark = benchmark.dropna()

    # Ensure both returns and benchmark have the same timezone
    returns.index = returns.index.tz_convert(None)
    benchmark.index = benchmark.index.tz_convert(None)
    
    CAGR = cagr(returns, period='daily', annualization=None)
    CAGR = round(CAGR * 100, 2)
    CAGR = CAGR.tolist()
    CAGR = str(CAGR) + "%"

    CUM = cum_returns(returns, starting_value=0, out=None) * 100
    CUM = CUM.iloc[-1]
    CUM = round(CUM, 2)
    CUM = CUM.tolist()
    CUM = str(CUM) + "%"

    VOL = qs.stats.volatility(returns, annualize=True)
    VOL = round(VOL * 100, 2)
    VOL = VOL.tolist()
    VOL = str(VOL) + "%"

    SR = qs.stats.sharpe(returns, rf=rf)
    SR = round(SR, 2)
    SR = SR.tolist()
    SR = str(SR)

    PortfolioAnalyser.SR = SR

    CR = qs.stats.calmar(returns)
    CR = round(CR, 2)
    CR = CR.tolist()
    CR = str(CR)

    PortfolioAnalyser.CR = CR

    STABILITY = stability_of_timeseries(returns)
    STABILITY = round(STABILITY, 2)
    STABILITY = str(STABILITY)

    MD = max_drawdown(returns, out=None)
    MD = round(MD * 100, 2)
    # MD = MD.tolist()
    MD = str(MD) + "%"

    SOR = sortino_ratio(returns, required_return=0, period='daily')
    SOR = round(SOR, 2)
    # SOR = SOR.tolist()
    SOR = str(SOR)

    SK = qs.stats.skew(returns)
    SK = round(SK, 2)
    # SK = SK.tolist()
    SK = str(SK)

    KU = qs.stats.kurtosis(returns)
    KU = round(KU, 2)
    # KU = KU.tolist()
    KU = str(KU)

    TA = tail_ratio(returns)
    TA = round(TA, 2)
    TA = str(TA)

    CSR = qs.stats.common_sense_ratio(returns)
    CSR = round(CSR, 2)
    # CSR = CSR.tolist()
    CSR = str(CSR)

    VAR = qs.stats.value_at_risk(
        returns, sigma=sigma_value, confidence=confidence_value
    )
    VAR = np.round(VAR, 2)
    VAR = VAR.tolist()
    VAR = str(VAR * 100) + " %"

    alpha, beta = alpha_beta(returns, benchmark, risk_free=rf)
    AL = round(alpha, 2)
    BTA = round(beta, 2)

    win_ratio = qs.stats.win_rate(returns)
    win_ratio = round(win_ratio * 100, 2)
    # win_ratio = win_ratio.tolist()
    win_ratio = str(win_ratio)

    IR = calculate_information_ratio(returns, benchmark.iloc[:, 0])
    IR = round(IR, 2)
    # IR = IR.tolist()
    IR = str(IR)

    ABSA = get_aspects(my_portfolio)

    data = {
        "": [
            "Annual return",
            "Cumulative return",
            "Annual volatility",
            "Winning day ratio",
            "Sharpe ratio",
            "Calmar ratio",
            "Information ratio",
            "Stability",
            "Max Drawdown",
            "Sortino ratio",
            "Skew",
            "Kurtosis",
            "Tail Ratio",
            "Common sense ratio",
            "Daily value at risk",
            "Alpha",
            "Beta",
            "Earnings Sentiment Score",
            "Revenue Sentiment Score",
            "Margins Sentiment Score",
            "Dividend Sentiment Score",
            "EBITDA Sentiment Score",
            "Debt Sentiment Score",
            "Overall Sentiment Score",
        ],
        "Backtest Results": [
            CAGR,
            CUM,
            VOL,
            f"{win_ratio}%",
            SR,
            CR,
            IR,
            STABILITY,
            MD,
            SOR,
            SK,
            KU,
            TA,
            CSR,
            VAR,
            AL,
            BTA,
            ABSA["Earnings"],
            ABSA["Revenue"],
            ABSA["Margins"],
            ABSA["Dividend"],
            ABSA["EBITDA"],
            ABSA["Debt"],
            ABSA["Sentiment"]
        ],
    }

    with open("data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Backtest"])
        writer.writerow(["Start date", my_portfolio.start_date])
        writer.writerow(["End date", my_portfolio.end_date])
        metrics = data[""]
        backtest = data["Backtest Results"]
        for i in range(len(metrics)):
            writer.writerow([metrics[i], backtest[i]])

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index("", inplace=True)
    df.style.set_properties(
        **{"background-color": "white", "color": "black", "border-color": "black"}
    )
    # display(df)

    PortfolioAnalyser.df = data

    my_color = np.where(returns >= 0, "blue", "grey")
    ret = plt.figure(figsize=(30, 8))
    plt.vlines(x=returns.index, ymin=0, ymax=returns, color=my_color, alpha=0.4)
    plt.title("Returns")

    PortfolioAnalyser.returns = returns
    PortfolioAnalyser.creturns = creturns
    PortfolioAnalyser.benchmark = benchmark
    PortfolioAnalyser.CAGR = CAGR
    PortfolioAnalyser.CUM = CUM
    PortfolioAnalyser.VOL = VOL
    PortfolioAnalyser.SR = SR
    PortfolioAnalyser.win_ratio = win_ratio
    PortfolioAnalyser.CR = CR
    PortfolioAnalyser.IR = IR
    PortfolioAnalyser.STABILITY = STABILITY
    PortfolioAnalyser.MD = MD
    PortfolioAnalyser.SOR = SOR
    PortfolioAnalyser.SK = SK
    PortfolioAnalyser.KU = KU
    PortfolioAnalyser.TA = TA
    PortfolioAnalyser.CSR = CSR
    PortfolioAnalyser.VAR = VAR
    PortfolioAnalyser.AL = AL
    PortfolioAnalyser.BTA = BTA
    PortfolioAnalyser.Earnings = ABSA["Earnings"]
    PortfolioAnalyser.Revenue = ABSA["Revenue"]
    PortfolioAnalyser.Margins = ABSA["Margins"]
    PortfolioAnalyser.Dividend = ABSA["Dividend"]
    PortfolioAnalyser.EBITDA = ABSA["EBITDA"]
    PortfolioAnalyser.Debt = ABSA["Debt"]
    PortfolioAnalyser.Sentiment = ABSA["Sentiment"]

    try:
        PortfolioAnalyser.orderbook = make_rebalance.output
    except Exception as e:
        OrderBook = pd.DataFrame(
            {
                "Assets": my_portfolio.portfolio,
                "Allocation": my_portfolio.weights,
            }
        )

        PortfolioAnalyser.orderbook = OrderBook.T

    wts = copy.deepcopy(my_portfolio.weights)
    indices = [i for i, x in enumerate(wts) if x == 0.0]

    while 0.0 in wts:
        wts.remove(0.0)

    for i in sorted(indices, reverse=True):
        del my_portfolio.portfolio[i]

    if not report:
      qs.plots.returns(returns, benchmark, cumulative=True)
      qs.plots.yearly_returns(returns, benchmark),
      qs.plots.monthly_heatmap(returns, benchmark)
      qs.plots.drawdown(returns)
      qs.plots.drawdowns_periods(returns)
      qs.plots.rolling_volatility(returns)
      qs.plots.rolling_sharpe(returns)
      qs.plots.rolling_beta(returns, benchmark)
      graph_opt(my_portfolio.portfolio, wts, pie_size=7, font_size=14)
      plot_sentiment(my_portfolio)

    else:
      qs.plots.returns(returns, benchmark, cumulative=True, savefig="retbench.png", show=False),
      qs.plots.yearly_returns(returns, benchmark, savefig="y_returns.png", show=False),
      qs.plots.monthly_heatmap(returns, benchmark, savefig="heatmap.png", show=False),
      qs.plots.drawdown(returns, savefig="drawdown.png", show=False),
      qs.plots.drawdowns_periods(returns, savefig="d_periods.png", show=False),
      qs.plots.rolling_volatility(returns, savefig="rvol.png", show=False),
      qs.plots.rolling_sharpe(returns, savefig="rsharpe.png", show=False),
      qs.plots.rolling_beta(returns, benchmark, savefig="rbeta.png", show=False),
      graph_opt(my_portfolio.portfolio, wts, pie_size=10, font_size=14, save=True)
      plot_sentiment(my_portfolio)
      pdf = FPDF()
      pdf.add_page()
      pdf.set_font("arial", "B", 14)
      pdf.image(
          "https://raw.githubusercontent.com/Armxyz1/Armxyz1/refs/heads/main/github-header-image.png",
          x=None,
          y=None,
          w=45,
          h=5,
          type="",
          link="https://github.com/Armxyz1",
      )
      pdf.cell(20, 15, f"Report", ln=1)
      pdf.set_font("arial", size=11)
      pdf.image("allocation.png", x=135, y=0, w=70, h=70, type="", link="")
      pdf.cell(20, 7, f"Start date: " + str(my_portfolio.start_date), ln=1)
      pdf.cell(20, 7, f"End date: " + str(my_portfolio.end_date), ln=1)
      ret.savefig("ret.png")

      pdf.cell(20, 7, f"", ln=1)
      pdf.cell(20, 7, f"Annual return: " + str(CAGR), ln=1)
      pdf.cell(20, 7, f"Cumulative return: " + str(CUM), ln=1)
      pdf.cell(20, 7, f"Annual volatility: " + str(VOL), ln=1)
      pdf.cell(20, 7, f"Winning day ratio: " + str(win_ratio), ln=1)
      pdf.cell(20, 7, f"Sharpe ratio: " + str(SR), ln=1)
      pdf.cell(20, 7, f"Calmar ratio: " + str(CR), ln=1)
      pdf.cell(20, 7, f"Information ratio: " + str(IR), ln=1)
      pdf.cell(20, 7, f"Stability: " + str(STABILITY), ln=1)
      pdf.cell(20, 7, f"Max drawdown: " + str(MD), ln=1)
      pdf.cell(20, 7, f"Sortino ratio: " + str(SOR), ln=1)
      pdf.cell(20, 7, f"Skew: " + str(SK), ln=1)
      pdf.cell(20, 7, f"Kurtosis: " + str(KU), ln=1)
      pdf.cell(20, 7, f"Tail ratio: " + str(TA), ln=1)
      pdf.cell(20, 7, f"Common sense ratio: " + str(CSR), ln=1)
      pdf.cell(20, 7, f"Daily value at risk: " + str(VAR), ln=1)
      pdf.cell(20, 7, f"Alpha: " + str(AL), ln=1)
      pdf.cell(20, 7, f"Beta: " + str(BTA), ln=1)
      pdf.cell(20, 7, f"Earnings Sentiment Score: " + str(ABSA["Earnings"]), ln=1)
      pdf.cell(20, 7, f"Revenue Sentiment Score: " + str(ABSA["Revenue"]), ln=1)
      pdf.cell(20, 7, f"Margins Sentiment Score: " + str(ABSA["Margins"]), ln=1)
      pdf.cell(20, 7, f"Dividend Sentiment Score: " + str(ABSA["Dividend"]), ln=1)
      pdf.cell(20, 7, f"EBITDA Sentiment Score: " + str(ABSA["EBITDA"]), ln=1)
      pdf.cell(20, 7, f"Debt Sentiment Score: " + str(ABSA["Debt"]), ln=1)
      pdf.cell(20, 7, f"Overall Sentiment Score: " + str(ABSA["Sentiment"]), ln=1)


      pdf.image("ret.png", x=-20, y=None, w=250, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("y_returns.png", x=None, y=None, w=200, h=100, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("retbench.png", x=None, y=None, w=200, h=100, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("heatmap.png", x=None, y=None, w=200, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("drawdown.png", x=None, y=None, w=200, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("d_periods.png", x=None, y=None, w=200, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("rvol.png", x=None, y=None, w=190, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("rsharpe.png", x=None, y=None, w=190, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("rbeta.png", x=None, y=None, w=190, h=80, type="", link="")
      pdf.cell(20, 7, f"", ln=1)
      pdf.image("sentiment.png", x=None, y=None, w=200, h=80, type="", link="")

      pdf.output(dest="F", name=filename)
    #   print("The PDF was generated successfully!")


def flatten(subject) -> list:
    muster = []
    for item in subject:
        if isinstance(item, (list, tuple, set)):
            muster.extend(flatten(item))
        else:
            muster.append(item)
    return muster


def graph_opt(my_portfolio, my_weights, pie_size, font_size, save=False):
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(pie_size, pie_size)
    ax1.pie(my_weights, labels=my_portfolio, autopct="%1.1f%%", shadow=False, colors=CS)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.rcParams["font.size"] = font_size
    if save:
      plt.savefig("allocation.png")
    # plt.show()


def normalize_weights(my_portfolio) -> list:
    total = sum(my_portfolio.weights)
    return [x / total for x in my_portfolio.weights]


def efficient_frontier(my_portfolio, perf=True) -> list:
    # changed to take in desired timeline, the problem is that it would use all historical data
    ohlc = yf.download(
        my_portfolio.portfolio,
        start=my_portfolio.start_date,
        end=my_portfolio.end_date,
        progress=False,
    )
    prices = ohlc["Adj Close"].dropna(how="all")
    df = prices.filter(my_portfolio.portfolio)

    # sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
    df = df.fillna(1)
    if my_portfolio.expected_returns == None:
        my_portfolio.expected_returns = 'mean_historical_return'
    if my_portfolio.risk_model == None:
        my_portfolio.risk_model = 'sample_cov'
    mu = expected_returns.return_model(df, method=my_portfolio.expected_returns)
    S = risk_models.risk_matrix(df, method=my_portfolio.risk_model)

    # optimize for max sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
    if my_portfolio.min_weights is not None:
        ef.add_constraint(lambda x: x >= my_portfolio.min_weights)
    if my_portfolio.max_weights is not None:
        ef.add_constraint(lambda x: x <= my_portfolio.max_weights)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    wts = cleaned_weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        pred = ef.portfolio_performance(verbose=True)

    return flatten(result)


def hrp(my_portfolio, perf=True) -> list:
    # changed to take in desired timeline, the problem is that it would use all historical data

    ohlc = yf.download(
        my_portfolio.portfolio,
        start=my_portfolio.start_date,
        end=my_portfolio.end_date,
        progress=False,
    )
    prices = ohlc["Adj Close"].dropna(how="all")
    prices = prices.filter(my_portfolio.portfolio)

    # sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
    prices = prices.fillna(1)

    rets = expected_returns.returns_from_prices(prices)
    hrp = HRPOpt(rets)
    hrp.optimize()
    weights = hrp.clean_weights()

    wts = weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        hrp.portfolio_performance(verbose=True)

    return flatten(result)


def mean_var(my_portfolio, vol_max=0.15, perf=True) -> list:
    # changed to take in desired timeline, the problem is that it would use all historical data

    ohlc = yf.download(
        my_portfolio.portfolio,
        start=my_portfolio.start_date,
        end=my_portfolio.end_date,
        progress=False,
    )
    prices = ohlc["Adj Close"].dropna(how="all")
    prices = prices.filter(my_portfolio.portfolio)

    # sometimes we will pick a date range where company isn't public we can't set price to 0 so it has to go to 1
    prices = prices.fillna(1)

    if my_portfolio.expected_returns == None:
        my_portfolio.expected_returns = 'capm_return'
    if my_portfolio.risk_model == None:
        my_portfolio.risk_model = 'ledoit_wolf'

    mu = expected_returns.return_model(prices, method=my_portfolio.expected_returns)
    S = risk_models.risk_matrix(prices, method=my_portfolio.risk_model)

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
    if my_portfolio.min_weights is not None:
        ef.add_constraint(lambda x: x >= my_portfolio.min_weights)
    if my_portfolio.max_weights is not None:
        ef.add_constraint(lambda x: x <= my_portfolio.max_weights)
    ef.efficient_risk(vol_max)
    weights = ef.clean_weights()

    wts = weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        ef.portfolio_performance(verbose=True)

    return flatten(result)


def min_var(my_portfolio, perf=True) -> list:
    ohlc = yf.download(
        my_portfolio.portfolio,
        start=my_portfolio.start_date,
        end=my_portfolio.end_date,
        progress=False,
    )
    prices = ohlc["Adj Close"].dropna(how="all")
    prices = prices.filter(my_portfolio.portfolio)

    if my_portfolio.expected_returns == None:
        my_portfolio.expected_returns = 'capm_return'
    if my_portfolio.risk_model == None:
            my_portfolio.risk_model = 'ledoit_wolf'

    mu = expected_returns.return_model(prices, method=my_portfolio.expected_returns)
    S = risk_models.risk_matrix(prices, method=my_portfolio.risk_model)

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=my_portfolio.diversification)
    if my_portfolio.min_weights is not None:
        ef.add_constraint(lambda x: x >= my_portfolio.min_weights)
    if my_portfolio.max_weights is not None:
        ef.add_constraint(lambda x: x <= my_portfolio.max_weights)
    ef.min_volatility()
    weights = ef.clean_weights()

    wts = weights.items()

    result = []
    for val in wts:
        a, b = map(list, zip(*[val]))
        result.append(b)

    if perf is True:
        ef.portfolio_performance(verbose=True)

    return flatten(result)


def optimize_portfolio(my_portfolio, vol_max=25, pie_size=5, font_size=14):
    if my_portfolio.optimizer == None:
        raise Exception("You didn't define any optimizer in your portfolio!")
    returns1 = get_returns(
        my_portfolio.portfolio,
        my_portfolio.weights,
        start_date=my_portfolio.start_date,
        end_date=my_portfolio.end_date,
    )
    creturns1 = (returns1 + 1).cumprod()

    port = copy.deepcopy(my_portfolio.portfolio)

    wts = [1.0 / len(my_portfolio.portfolio)] * len(my_portfolio.portfolio)

    optimizers = {
        "EF": efficient_frontier,
        "MEANVAR": mean_var,
        "HRP": hrp,
        "MINVAR": min_var,
    }
    
    if my_portfolio.optimizer in optimizers.keys():
        if my_portfolio.optimizer == "MEANVAR":
            wts = optimizers.get(my_portfolio.optimizer)(my_portfolio, my_portfolio.max_vol)
        else:
            wts = optimizers.get(my_portfolio.optimizer)(my_portfolio)
    else:
        opt = my_portfolio.optimizer
        my_portfolio.weights = opt()

    print("\n")

    indices = [i for i, x in enumerate(wts) if x == 0.0]

    while 0.0 in wts:
        wts.remove(0.0)

    for i in sorted(indices, reverse=True):
        del port[i]

    graph_opt(port, wts, pie_size, font_size)

    print("\n")

    returns2 = get_returns(
        port, wts, start_date=my_portfolio.start_date, end_date=my_portfolio.end_date
    )
    creturns2 = (returns2 + 1).cumprod()

    plt.rcParams["font.size"] = 13
    plt.figure(figsize=(30, 10))
    plt.xlabel("Portfolio vs Benchmark")

    ax1 = creturns1.plot(color="blue", label="Without optimization")
    ax2 = creturns2.plot(color="red", label="With optimization")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    plt.legend(l1 + l2, loc=2)
    plt.show()


def check_schedule(rebalance) -> bool:
    valid_schedule = False
    if rebalance.lower() in rebalance_periods.keys():
        valid_schedule = True
    return valid_schedule


def valid_range(start_date, end_date, rebalance) -> tuple:

    # make the start date to a datetime
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")

    # custom dates don't need further chekings
    if type(rebalance) is list:
        return start_date, rebalance[-1]
    
    # make the end date to a datetime
    end_date = dt.datetime.strptime(str(end_date), "%Y-%m-%d")

    # gets the number of days
    days = (end_date - start_date).days

    # checking that date range covers rebalance period
    if rebalance in rebalance_periods.keys() and days <= (int(rebalance_periods[rebalance])):
        raise KeyError("Date Range does not encompass rebalancing interval")

    # we will needs these dates later on so we'll return them back
    return start_date, end_date


def get_date_range(start_date, end_date, rebalance) -> list:
    # this will keep track of the rebalancing dates and we want to start on the first date
    rebalance_dates = [start_date]
    input_date = start_date

    if rebalance in rebalance_periods.keys():
        # run for an arbitrarily large number we'll resolve this by breaking when we break the equality
        for i in range(1000):
            # increment the date based on the selected period
            input_date = input_date + dt.timedelta(days=rebalance_periods.get(rebalance))
            if input_date <= end_date:
                # append the new date if it is earlier or equal to the final date
                rebalance_dates.append(input_date)
            else:
                # break when the next rebalance date is later than our end date
                break

    # then we want to return those dates
    return rebalance_dates

def get_aspects(my_portfolio) -> dict:
    obj = interface.NewsDatabase()
    stocks = my_portfolio.portfolio
    weights = normalize_weights(my_portfolio)
    results = {}
    for aspect in aspects:
        results[aspect] = 0
    for i in range(len(stocks)):
        stock = stocks[i]
        weight = weights[i]
        values = obj.get_values_by_ticker(stock)
        for aspect in aspects:
            results[aspect] += weight * values[aspect]
    return results

def get_NIFTY_sentiment() -> float:
    NIFTY_50_ratios = {
        "ITC.NS" : 3.03,
        "TCS.NS" : 4.91,
        "ICICIBANK.NS" : 6.90,
        "HDFCBANK.NS" : 8.10,
        "KOTAKBANK.NS" : 3.51,
        "BAJAJFINSV.NS" : 1.20,
        "RELIANCE.NS" : 12.86,
        "INFY.NS" : 7.66,
        "HINDUNILVR.NS" : 2.67,
        "AXISBANK.NS" : 2.57,
        "SBIN.NS" : 2.54,
        "BAJFINANCE.NS" : 2.37,
        "LT.NS" : 2.74,
        "MARUTI.NS" : 1.37,
        "TATASTEEL.NS" : 1.37,
        "SUNPHARMA.NS" : 1.34,
        "M&M.NS" : 1.18,
        "TECHM.NS" : 1.05,
        "NESTLEIND.NS" : 0.87,
        "ONGC.NS" : 0.78,
        "DIVISLAB.NS" : 0.77,
        "ASIANPAINT.NS" : 1.95,
        "BHARTIARTL.NS" : 2.33,
        "HCLTECH.NS" : 1.53,
        "TITAN.NS" : 1.37,
        "POWERGRID.NS" : 1.04,
        "COALINDIA.NS" : 0.51,
        "WIPRO.NS" : 1.01,
        "NTPC.NS" : 0.99,
        "ULTRACEMCO.NS" : 1.02,
        "EICHERMOT.NS" : 0.49,
        "INDUSINDBK.NS" : 0.85,
        "HINDALCO.NS" : 0.94,
        "TATAMOTORS.NS" : 1.05,
        "JSWSTEEL.NS" : 0.94,
        "BPCL.NS" : 0.46,
        "HDFCLIFE.NS" : 0.72,
        "SHREECEM.NS" : 0.46,
        "GRASIM.NS" : 0.85,
        "BAJAJ-AUTO.NS" : 0.65,
        "ADANIPORTS.NS" : 0.82,
        "CIPLA.NS" : 0.68,
        "DRREDDY.NS" : 0.67,
        "TATACONSUM.NS" : 0.66,
        "HDFCBANK.NS" : 8.10,
        "APOLLOHOSP.NS" : 0.61,
        "UPL.NS" : 0.60,
        "SBILIFE.NS" : 0.65,
        "HEROMOTOCO.NS" : 0.43,
        "BRITANNIA.NS" : 0.52,
    }

    sentiment = 0
    obj = interface.NewsDatabase()
    for stock in NIFTY_50_ratios.keys():
        sentiment += obj.get_values_by_ticker(stock)["Sentiment"] * (NIFTY_50_ratios[stock] / 100)
    return sentiment

def plot_sentiment(my_portfolio) -> None:
    sentiment = get_aspects(my_portfolio)["Sentiment"]
    nifty_sentiment = get_NIFTY_sentiment()
    # Plot a bar graph comparing sentiment and NIFTY sentiment
    plt.figure(figsize=(10, 5))
    plt.bar(["Portfolio", "NIFTY 50"], [sentiment, nifty_sentiment], color=["#4D8BBB", "#F9DE86"])
    plt.title("Sentiment Analysis")
    plt.ylabel("Sentiment Score")
    plt.savefig("sentiment.png")
    pass
    
def make_rebalance(
    start_date,
    end_date,
    optimize,
    portfolio_input,
    rebalance,
    allocation,
    vol_max,
    div,
    min,
    max,
    expected_returns,
    risk_model,
) -> pd.DataFrame:
    sdate = str(start_date)[:10]
    if rebalance[0] != sdate:

        # makes sure the start date matches the first element of the list of custom rebalance dates
        if type(rebalance) is list:
            raise KeyError("the rebalance dates and start date doesn't match")

        # makes sure that the value passed through for rebalancing is a valid one
        valid_schedule = check_schedule(rebalance)

        if valid_schedule is False:
            raise KeyError("Not an accepted rebalancing schedule")

    # this checks to make sure that the date range given works for the rebalancing
    start_date, end_date = valid_range(start_date, end_date, rebalance)

    # this function will get us the specific dates
    if rebalance[0] != sdate:
        dates = get_date_range(start_date, end_date, rebalance)
    else:
        dates = rebalance

    # we are going to make columns with the end date and the weights
    columns = ["end_date"] + portfolio_input

    # then make a dataframe with the index being the tickers
    output_df = pd.DataFrame(index=portfolio_input)

    for i in range(len(dates) - 1):

        try:
            portfolio = Engine(
                start_date=dates[0],
                end_date=dates[i + 1],
                portfolio=portfolio_input,
                weights=allocation,
                optimizer="{}".format(optimize),
                max_vol=vol_max,
                diversification=div,
                min_weights=min,
                max_weights=max,
                expected_returns=expected_returns,
                risk_model=risk_model,
            )

        except TypeError:
            portfolio = Engine(
                start_date=dates[0],
                end_date=dates[i + 1],
                portfolio=portfolio_input,
                weights=allocation,
                optimizer=optimize,
                max_vol=vol_max,
                diversification=div,
                min_weights=min,
                max_weights=max,
                expected_returns=expected_returns,
                risk_model=risk_model,
            )

        output_df["{}".format(dates[i + 1])] = portfolio.weights

    # we have to run it one more time to get what the optimization is for up to today's date
    try:
        portfolio = Engine(
            start_date=dates[0],
            portfolio=portfolio_input,
            weights=allocation,
            optimizer="{}".format(optimize),
            max_vol=vol_max,
            diversification=div,
            min_weights=min,
            max_weights=max,
            expected_returns=expected_returns,
            risk_model=risk_model,
        )

    except TypeError:
        portfolio = Engine(
            start_date=dates[0],
            portfolio=portfolio_input,
            weights=allocation,
            optimizer=optimize,
            max_vol=vol_max,
            diversification=div,
            min_weights=min,
            max_weights=max,
            expected_returns=expected_returns,
            risk_model=risk_model,
        )

    output_df["{}".format(TODAY)] = portfolio.weights

    make_rebalance.output = output_df
    print("Rebalance schedule: ")
    print(output_df)
    return output_df