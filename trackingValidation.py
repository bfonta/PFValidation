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

import collections

import argparse

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ValidPlotter:
    def __init__(self, tag, odir):
        self.markers = ("s", "v", "o", "x", "^")
        self.colors = ("blue", "orange", "green", "red", "purple")
        self.fontsize = 40
        plt.rcParams.update({'font.size': 22})

        if odir:
            self.savedir = odir
        else:
            self.savedir = os.getcwd() + "/Plots_" + tag + "/"
        makedir(os.path.join(self.savedir))

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
                   ylabel="", xlabel="", title="",
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
        ax1.set_xlabel('', **sizeargs)

        if len(histos) > 1:
            ax2.set_ylabel('Ratio', **sizeargs)
            ax2.set_xlabel(xlabel, **sizeargs)
            ax2.set_ylim(ratio_ylim)
            ax2.hlines(y=1., xmin=hratio.axes[0].edges[0], xmax=hratio.axes[0].edges[-1],
                    linewidth=2, linestyle='--', color='gray')

            if modify_ticks:
                ax2.tick_params(axis='x', which='major', labelsize=0.4*self.fontsize, rotation=15)

        ax1.legend()

        hep.cms.text(' Preliminary', fontsize=self.fontsize, ax=ax1)
        hep.cms.lumitext(title, fontsize=0.8*self.fontsize, ax=ax1)
        hep.style.use("CMS")

        for ext in ('.png', '.pdf'):
            plt.savefig(self.savedir + "{}.png".format(savename))

def getEfficiency(passing, total):
    yEff = []
    yEffErrUp = []
    yEffErrLow = []
    for yPass, yTot in zip(passing, total):
        if yTot>0:
            # error calculation with eff
            result = binomtest(k=int(yPass), n=int(yTot))
            yEff.append(result.statistic)
            yEffErrLow.append(result.proportion_ci(0.683).low)
            yEffErrUp.append(result.proportion_ci(0.683).high)
        else:
            yEff.append(0)
            yEffErrLow.append(0)
            yEffErrUp.append(0)
    return np.array(yEff), np.array(yEffErrLow), np.array(yEffErrUp)

class Validation(ABC):
    """
    Class to manage the validation.
    """
    def __init__(self, files, labels, tag, odir, directory):
        self.tag, self.odir = tag, odir
        
        self.dqm_paths, self.dqm_files = {}, {}
        for file, label in zip(files, labels):
            self.dqm_paths[label] = file
            self.dqm_files[label] = uproot.open(file)[directory]

        self.en_str = r"E ($\GeV$)"
        self.pt_str = r"p$_{\text{T}}$"
        self.eta_str = r"$\eta$"
        self.phi_str = r"$\phi$"


    def run(self, names, names_nested):
        """
        Plot level 0 histograms.
        """     
        histos = collections.defaultdict(dict)
        for label in self.dqm_paths.keys():
            for name in names:
                histos[label].update({name: self.dqm_files[label][name].to_hist()})
     
        plotter = ValidPlotter(tag, odir)
        for name in names:
            v_histos = [histos[label][name] for label in dqm_paths.keys()]
            v_labels = [label for label in dqm_paths.keys()]
            plotter.plotHistos(histos=v_histos,
                               labels=v_labels,
                               modify_ticks=True,
                               savename=name)

        histos_nested = collections.defaultdict(lambda: collections.defaultdict(dict))
        for label in dqm_paths.keys():
            for coll in self.collections.values():
                for name in names_nested:
                    histos_nested[label][coll].update({name: self.dqm_files[label][coll][name].to_hist()})
     
        for coll in self.collections.values():
            makedir(os.path.join(plotter.savedir, coll))
            for name, labels in names_nested.items():
                v_histos = [histos_nested[label][coll][name] for label in self.dqm_paths.keys()]
                v_labels = [label for label in self.dqm_paths.keys()]
                plotter.plotHistos(histos=v_histos,
                                   labels=v_labels,
                                   ylabel=labels[0],
                                   xlabel=labels[1],
                                   title=coll,
                                   savename=coll + '/' + name)

class HGCalValidation(Validation):
    def __init__(self, files, labels, tag, odir):
        directory = "DQMData/Run 1/HLT/Run summary/HGCAL/HGCalValidator/hltTiclCandidate"
        super().__init__(files, labels, tag, odir, directory)

        self.collections = {
            "Electrons"           : "electrons", 
            "Photons"             : "photons", 
            "Muons"               : "muons", 
            "Pi0"                 : "neutral_pions", 
            "ChargedHadrons"      : "charged_hadrons", 
            "NeutralHadrons"      : "neutral_hadrons", 
        }

        names = ("Candidates PDG Id", "Candidates charge", "Candidates pT", 
                 "Candidates raw energy", "Candidates regressed energy", "Candidates type",
                 "N of tracksters in candidate")

        names, names_nested = self.define_names()
        super.run(names, names_nested)

    def define_histo_names(self):
        axes = {
            "energy": "E (GeV)",
            "pt": r"p$_{\text{T}}$",
            "eta": r"$\eta$",
            "phi": r"$\phi$",
        }
        metrics = {
            "eff": "Efficiency",
            "fake": "Fake Rate",
        }

        names_nested = collections.defaultdict(lambda: collections.defaultdict(dict))
        for coll in self.collections.values():
            for metric, ylabel in metrics.items():
                for step in ["energy", "pid"]:
                    for axis, xlabel in axes.items():
                        names_nested_coll[f"{metric}_{coll}_{step}_{axis}"] = (ylabel, xlabel)
                # Only include "track" variables for charged particles
                if coll in ["electrons", "muons", "charged_hadrons"]:
                    for axis, xlabel in axes.items():
                        names_nested_coll[f"{metric}_{coll}_track_{axis}"] = (ylabel, xlabel)
            
        names_nested[coll] = names_nested_coll
        return names_nested

        
class TrackValidation(Validation):
    """
    Produces the tracking validation comparison plots.
    """
    def __init__(self, files, labels, tag, odir):
        directory = "DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTtp"
        super().__init__(files, labels, tag, odir, directory)

        self.collections = {
        "GeneralTracks"           : "hltGeneral_hltAssociatorByHits", 
        "PixelTracks"             : "hltPhase2Pixel_hltAssociatorByHits", 
        # "InitialStepTracks"       : "hltInitialStepTrackSelectionHighPurity_hltAssociatorByHits", 
        # "HighPtTripletStepTracks" : "hltHighPtTripletStepTrackSelectionHighPurity_hltAssociatorByHits", 
        # "MergedSeeds"             : "hltMergedPixelHighPtTripletSeeds_hltAssociatorByHits"
        }

        names, names_nested = self.define_names()
        super.run(names, names_nested)

    def define_names(self):
        names = ("globalEfficiencies", "effic_vs_coll",
                 "fakerate_vs_coll", "pileuprate_coll",
                 "num_assoc(simToReco)_coll", "num_assoc(recoToSim)_coll")
        
        names_nested = {
            "effic_vs_dz"        : ("Efficiency", "dz"),
            "effic_vs_dxy"       : ("Efficiency", "dR"),
            "effic_vs_hit"       : ("Efficiency", "Hit"),
            "effic_vs_phi"       : ("Efficiency", phi_str),
            "effic"              : ("Efficiency", eta_str),
            "efficPt"            : ("Efficiency", pt_str),
            "fakerate_vs_dz"     : ("Fake Rate", "dz"),
            "fakerate_vs_dxy"    : ("Fake Rate", "dxy"),
            "fakerate_vs_dr"     : ("Fake Rate", "dR"),
            "fakerate_vs_hit"    : ("Fake Rate", "Hit"),
            "fakerate_vs_phi"    : ("Fake Rate", phi_str),
            "fakeratePt"         : ("Fake Rate", pt_str),
            "duplicatesRate_dz"  : ("Duplicate Rate", "dz"),
            "duplicatesRate_dxy" : ("Duplicate Rate", "dxy"),
            "duplicatesRate_dr"  : ("Duplicate Rate", "dR"),
            "duplicatesRate_hit" : ("Duplicate Rate", "Hit"),
            "duplicatesRate_phi" : ("Duplicate Rate", phi_str),
            "duplicatesRate_Pt"  : ("Duplicate Rate", pt_str)
        }

        return names, names_nested

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make track validation plots.')
    parser.add_argument('--files', nargs='+', type=str, required=True, help='Paths to the ROOT files.')
    parser.add_argument('--labels', nargs='+', type=str, required=False, help='Legend labels.')
    parser.add_argument('--tag', type=str, default=None, required=True, help='Tag to uniquely identify the plots.')
    parser.add_argument('--odir', type=str, required=False, help='Path to the output directory (if not specified, save to current directory).')
 = parser.parse_args()

    labels = args.labels if args.labels else [f"File_{i}" for i in range(len(args.files))]    
    assert len(args.files) == len(labels), "Number of files and labels must match."

    valid_kwargs = dict(files=files, labels=labels, tag=args.tag, odir=args.odir)
    trackReleaseValidation(**valid_kwargs)
    hgcalReleaseValidation(**valid_kwargs)
