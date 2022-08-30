from . import utils
from . import synthetic_population
from . import posterior
from . import analyses
import matplotlib as _mpl

_mpl.rcParams['figure.figsize']= (3.3, 2.5)
_mpl.rcParams['figure.dpi']= 300
_mpl.rcParams['axes.labelsize']= 7
_mpl.rcParams['xtick.labelsize']= 7
_mpl.rcParams['ytick.labelsize']= 7
_mpl.rcParams['legend.fontsize']= 7
_mpl.rcParams['font.size']= 7
_mpl.rcParams['font.family']= 'sans-serif'
_mpl.rcParams['font.sans-serif']= ['DejaVu Sans', 'Arial', 'Helvetica', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Avant Garde', 'sans-serif']
_mpl.rcParams['mathtext.fontset']='dejavusans'
_mpl.rcParams['axes.linewidth']= 0.5
_mpl.rcParams['grid.linewidth']= 0.5
_mpl.rcParams['lines.linewidth']= 1.
_mpl.rcParams['lines.markersize']= 3.
_mpl.rcParams['savefig.bbox']= 'tight'
_mpl.rcParams['savefig.pad_inches']= 0.01
#mpl.rcParams['text.latex.preamble']= '\usepackage{amsmath, amssymb, sfmath}'
