# coding: utf-8

_all_ = [ 'trackValidation.py' ]

import os
from abc import ABC, abstractmethod

import numpy as np
import uproot
import awkward as ak
import hist

import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep

import collections as coll

import argparse

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Plotter:
    def __init__(self, odir):
        self.markers = ("s", "v", "o", "x", "^")
        self.colors = ("blue", "orange", "green", "red", "purple")
        self.fontsize = 40
        plt.rcParams.update({'font.size': 22})

        if os.path.exists(odir):
            raise RuntimeError(f'Folder {odir} already exists!')
        
        self.savedir = odir
        makedir(self.savedir)

    def _div(self, num, den):
        """Ignore division by zero; they are correctly handled by the plot."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = num / den
        return ratio

    def ratioHist(self, num, den):
        hratio = num.copy()
        upvals, upvars = num.values(), num.variances()
        dovals, dovars = den.values(), den.variances()
        
        hratio.values()[:] = self._div(upvals, dovals)
        hratio.variances()[:] = np.abs(self._div(upvals, dovals))
        hratio.variances()[:] *= np.sqrt(self._div(upvars, upvals)**2 +  self._div(dovars, dovals)**2)

        return hratio
        
    def plotHistos(self, histos, labels,
                   savename,
                   ylabel="", xlabel="",
                   modify_ticks=False,
                   xlim=(None,None), ylim=(None,None), ratio_ylim=(0.5, 1.5),
                   yscale=None):
        """Plot histograms."""

        if any(sum(h.values()) for h in histos):
            print(f"All histograms in {savename} are empty. Skipping plot.")
            return

        # Create a figure
        plt.close()
        if len(histos) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,16),
                                        gridspec_kw={'height_ratios': [3, 1]})
            plt.subplots_adjust(wspace=0, hspace=0.05)        
        else:
            fig, ax1 = plt.subplots(figsize=(20, 16))
            ax2 = None

        colors = plt.cm.tab10.colors
        plot_args = dict(linewidth=4)

        # Plot the main histograms
        for idx, (h, label) in enumerate(zip(histos, labels)):
            color = colors[idx % len(colors)]
            h.plot(ax=ax1, label=label, color=color, **plot_args)

        # Plot ratios with respect to the first histogram
        reference_hist = histos[0]
        for idx, h in enumerate(histos[1:], start=1):
            hratio = self.ratioHist(reference_hist, h)
            color = colors[idx % len(colors)]
            hratio.plot(ax=ax2, color=color, histtype='errorbar', markersize=0.5*self.fontsize, label=f"{labels[idx]}/" + labels[0])

        sizeargs = dict(fontsize=0.7*self.fontsize)
        ax1.set_ylabel(ylabel, loc="top", **sizeargs)

        if len(histos) > 1:
            ax1.set_xlabel('', **sizeargs)
            ax2.set_ylabel('Ratio', **sizeargs)
            ax2.set_xlabel(xlabel, **sizeargs)
            ax2.set_ylim(ratio_ylim)
            ax2.hlines(y=1., xmin=hratio.axes[0].edges[0], xmax=hratio.axes[0].edges[-1],
                       linewidth=2, linestyle='--', color='gray')

            if modify_ticks:
                ax2.tick_params(axis='x', which='major', labelsize=0.4*self.fontsize, rotation=15)
        else:
            ax1.set_xlabel(xlabel, **sizeargs)
            
        ax1.legend()

        hep.style.use("CMS")
        hep.cms.text(' Preliminary', fontsize=self.fontsize, ax=ax1)
        # hep.cms.lumitext(title, fontsize=0.8*self.fontsize, ax=ax1)

        for ext in ('.png', '.pdf'):
            makedir(os.path.join(self.savedir, os.path.dirname(savename)))
            savename = savename.replace(' ', '_').replace('.', 'p').replace('-', 'm')
            plt.savefig(self.savedir + "{}.png".format(savename))
            

class ValidationAbstract(ABC):
    """
    Class to manage the validation.
    """
    def __init__(self, files, labels, odir, directory):
        self.odir = odir
        
        self.dqm_paths, self.dqm_files = {}, {}
        for file, label in zip(files, labels):
            self.dqm_paths[label] = file
            self.dqm_files[label] = uproot.open(file)[directory]

        self.en_str = r"E [GeV]"
        self.pt_str = r"p$_{\text{T}}$ [GeV]"
        self.eta_str = r"$\eta$"
        self.phi_str = r"$\phi$"

    @abstractmethod
    def define_names(self) -> dict:
        """
        The user should create a dictionary where the keys represent the path of the histograms (added on top of the `directory` argument),
        and the values are the y and x axis labels.
        """
        pass
    
    def run(self, names) -> None:
        """
        Plot level 0 histograms.
        """     
        plotter = Plotter(self.odir)
        for name, labels in names.items():
            v_histos = [self.dqm_files[key][name].to_hist() for key in self.dqm_paths.keys()]
            v_labels = [label for label in self.dqm_paths.keys()]
            plotter.plotHistos(histos=v_histos,
                               labels=v_labels,
                               ylabel=labels[0],
                               xlabel=labels[1],
                               savename=name)

class HGCalValidation(ValidationAbstract):
    def __init__(self, files, labels, odir):
        directory = "DQMData/Run 1/HLT/Run summary/HGCAL/HGCalValidator/hltTiclCandidate"
        super().__init__(files, labels, odir, directory)

        names = self.define_names()
        super().run(names)

    def define_names(self):

        axes = {
            "energy" : self.en_str,
            "pt"     : self.pt_str,
            "eta"    : self.eta_str,
            "phi"    : self.phi_str,
        }
        metrics = {
            "eff"  : "Efficiency",
            "fake" : "Fake Rate",
        }

        folders = ("electrons", "photons", "muons", "neutral_pions", "charged_hadrons", "neutral_hadrons")

        names = coll.defaultdict(lambda: coll.defaultdict(dict))

        names["Candidates PDG Id"]            = ("", "PDF Id")
        names["Candidates charge"]            = ("", "Charge")
        names["Candidates pT"]                = ("", self.pt_str)
        names["Candidates raw energy"]        = ("", self.en_str)
        names["Candidates regressed energy"]  = ("", self.en_str)
        names["Candidates type"]              = ("", "Type")
        names["N of tracksters in candidate"] = ("", "Multiplicity")

        for fld in folders:
            for metric, ylabel in metrics.items():
                for step in ["energy", "pid"]:
                    for axis, xlabel in axes.items():
                        names[f"{fld}/{metric}_{fld}_{step}_{axis}"] = (ylabel, xlabel)
                        # Only include "track" variables for charged particles
                if fld in ["electrons", "muons", "charged_hadrons"]:
                    for axis, xlabel in axes.items():
                        names[f"{fld}/{metric}_{fld}_track_{axis}"] = (ylabel, xlabel)

        return names

class TrackValidation(ValidationAbstract):
    """
    Produces the tracking validation comparison plots.
    """
    def __init__(self, files, labels, odir):
        directory = "DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTtp"
        super().__init__(files, labels, odir, directory)

        names = self.define_names()
        super.run(names)

    def define_names(self):

        folders = ("hltGeneral_hltAssociatorByHits", "hltPhase2Pixel_hltAssociatorByHits", "hltInitialStepTrackSelectionHighPurity_hltAssociatorByHits", 
                   # "hltHighPtTripletStepTrackSelectionHighPurity_hltAssociatorByHits", "hltMergedPixelHighPtTripletSeeds_hltAssociatorByHits"
                   )

        names_dict = {
            "effic_vs_dz"        : ("Efficiency", "dz"),
            "effic_vs_dxy"       : ("Efficiency", "dR"),
            "effic_vs_hit"       : ("Efficiency", "Hit"),
            "effic_vs_phi"       : ("Efficiency", self.phi_str),
            "effic"              : ("Efficiency", self.eta_str),
            "efficPt"            : ("Efficiency", self.pt_str),
            "fakerate_vs_dz"     : ("Fake Rate", "dz"),
            "fakerate_vs_dxy"    : ("Fake Rate", "dxy"),
            "fakerate_vs_dr"     : ("Fake Rate", "dR"),
            "fakerate_vs_hit"    : ("Fake Rate", "Hit"),
            "fakerate_vs_phi"    : ("Fake Rate", self.phi_str),
            "fakeratePt"         : ("Fake Rate", self.pt_str),
            "duplicatesRate_dz"  : ("Duplicate Rate", "dz"),
            "duplicatesRate_dxy" : ("Duplicate Rate", "dxy"),
            "duplicatesRate_dr"  : ("Duplicate Rate", "dR"),
            "duplicatesRate_hit" : ("Duplicate Rate", "Hit"),
            "duplicatesRate_phi" : ("Duplicate Rate", self.phi_str),
            "duplicatesRate_Pt"  : ("Duplicate Rate", self.pt_str)
        }

        names_nested = coll.defaultdict(lambda: coll.defaultdict(dict))

        names["globalEfficiencies"]        = ("", "Collection")
        names["effic_vs_coll"]             = ("", "Collection")
        names["fakerate_vs_coll"]          = ("", "Collection")
        names["pileuprate_coll"]           = ("", "Collection")
        names["num_assoc(simToReco)_coll"] = ("", "Collection")
        names["num_assoc(recoToSim)_coll"] = ("", "Collection")

        for fld in folders:
            for key, value in names_dict.items():
                names[fld + '/' + key] = value
                
        return names


class Validation:
    """
    Factory for the various validation classes.
    """
    valid_types = {
      "tracker": TrackValidation,
      "hgcal": HGCalValidation,
    }
    
    @staticmethod
    def validate(valid_type, class_kwargs):
        valid_class = Validation.valid_types.get(valid_type)
        if valid_class is None:
            raise ValueError(f"Invalid validation type: {valid_type}. Options are {list(Validation.valid_types.keys())}.")
        return valid_class(**class_kwargs)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make track validation plots.')
    parser.add_argument('--files', nargs='+', type=str, required=True, help='Paths to the ROOT files.')
    parser.add_argument('--labels', nargs='+', type=str, required=False, help='Legend labels.')
    parser.add_argument('--odir', type=str, required=True, help='Path to the output directory (if not specified, save to current directory).')
    args = parser.parse_args()

    labels = args.labels if args.labels else [f"File_{i}" for i in range(len(args.files))]    
    assert len(args.files) == len(labels), "Number of files and labels must match."

    valid_kwargs = dict(files=args.files, labels=labels, odir=args.odir)

    for key in ('hgcal',): # 'tracker'
        Validation.validate(key, valid_kwargs)

