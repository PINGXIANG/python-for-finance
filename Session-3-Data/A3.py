# import yahoo_fin.stock_info as si
# import numpy as np
# import seaborn as sns
# import sys
# import bokeh
#
# df = si.get_data('BILI')
# # log_return = np.log(df['open']) - np.log(df['open'].shift(1))
# #
# # sns.histplot(log_return)
#
# tickerDict = {}
# listTickers = si.tickers_sp500()
# for listTicker in listTickers:
#     try:
#         tickerDict[listTicker] = si.get_data(listTicker)
#         print(listTicker, "collected")
#     except:
#         e = sys.exc_info()[0]
#         tickerDict[listTicker] = df[0:0]
#         print(listTicker, 'falled')
#         print(Exception)
#
from scipy.stats import norm
import numpy as np
import bokeh.plotting.figure as bk_figure
from bokeh.io import curdoc, show
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.io import output_notebook  # enables plot interface in J notebook

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler


def BSCall(St, K, sigma, T, t=0, r=0):
    """
    #Inputs:
    S0: initial stock price
    K: strike
    sigma: instantaneous volatility
    T: time to maturity
    #Output:
    Black-Scholes Call price
    """

    deltaT = T - t
    sigmaT = np.multiply(sigma, np.sqrt(deltaT))
    lnSK = np.log(St / K)
    d1 = (lnSK + (r + (np.power(sigma, 2.)) / 2.)) / sigmaT
    d2 = d1 - sigmaT

    return np.multiply(St, norm.cdf(d1)) - np.multiply(K, norm.cdf(d2)) * np.exp(-r * deltaT)


# Set up data
S0, K, sigma, t, T = 100., 100., 0.2, 0., 1.

n = 2000
KK = np.linspace(0.01, 10.0)
BSCall_vec = np.vectorize(BSCall)
CC = BSCall_vec(S0, KK, sigma, T)
source = ColumnDataSource(data=dict(x=KK, y=CC))

# Set up plot
plot = bk_figure(plot_height=400, plot_width=400, title="Patrick's BSCall Function",
                 tools="crosshair,pan,reset,save,wheel_zoom",
                 x_range=[np.min(KK), np.max(KK)], y_range=[np.min(CC), np.max(CC)])

plot.line('Strike K', 'Call Price C0', source=source, line_width=3, line_alpha=0.6)

# Set up widgets
text = TextInput(title="title", value='BSCall Function')
TT = Slider(title="Maturity T", value=sigma, start=0.1, end=10.0, step=0.5)
sigmas = Slider(title="Sigma", value=T, start=0.01, end=10., step=0.01)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value


def update_data(attrname, old, new):
    # Get the current slider values
    T = TT.value
    sigma = sigmas.value

    # Generate the new curve
    new_CC = BSCall_vec(S0, KK, sigma, T)

    source.data = dict(x=KK, y=new_CC)


for w in [TT, sigmas]:
    w.on_change('value', update_data)  # TODO: What is on change?

# Set up layouts and add to document
inputs = widgetbox(text, TT, sigmas)
layout = row(plot, widgetbox(text, TT, sigmas))


def modify_doc(doc):
    doc.add_root(row(layout, width=800))
    doc.title = "Sliders"
    text.on_change('value', update_title)


handler = FunctionHandler(modify_doc)
show(Application(handler))
