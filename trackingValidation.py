# coding: utf-8

_all_ = [ 'trackValidation.py' ]

import os
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
        
    def plotHistos(self, h1, h2,
                   savename,
                   label1="h1", label2="h2",
                   ylabel="", xlabel="", title="",
                   modify_ticks=False,
                   xlim=(None,None), ylim=(None,None), ratio_ylim=(0.5, 1.5),
                   yscale=None):
        """Plot histograms."""

        # Create a figure
        plt.close()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,16),
                                       gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(wspace=0, hspace=0.05)
        
        if float(sum(h2.values())) == 0.0:
            print("Histogram {} only has zeros!".format(h2.name))
            return

        plot_args = dict(linewidth=4)
        artist1 = h1.plot(ax=ax1, label=label1, color='blue',  **plot_args)
        artist2 = h2.plot(ax=ax1, label=label2, color='orange', **plot_args)
        
        hratio = self.ratioHist(h1, h2)
        artist_ratio = hratio.plot(ax=ax2, color='black', histtype='errorbar',
                                   markersize=0.5*self.fontsize)

        sizeargs = dict(fontsize=0.7*self.fontsize)
        ax1.set_ylabel(ylabel, loc="top", **sizeargs)
        ax1.set_xlabel('', **sizeargs)
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

def hgcalReleaseValidation(filepath1, filepath2, tag, odir):
    """
    Produces the HGCAL validation comparison plots.
    """
    dqm_paths = {"pre3": filepath1,
                 "pre4": filepath2}

    dqm_files = {"pre3": uproot.open(dqm_paths['pre3'])["DQMData/Run 1/HLT/Run summary/HGCAL/HGCalValidator/hltTiclCandidate"],
                 "pre4": uproot.open(dqm_paths['pre4'])["DQMData/Run 1/HLT/Run summary/HGCAL/HGCalValidator/hltTiclCandidate"]}

    hgcalCollections = {
        "Electrons"           : "electrons", 
        "Photons"             : "photons", 
        "Muons"               : "muons", 
        "Pi0"                 : "neutral_pions", 
        "ChargedHadrons"      : "charged_hadrons", 
        "NeutralHadrons"      : "neutral_hadrons", 
    }

    # Plot level 0 histograms
    names_level0 = [ "Candidates PDG Id", "Candidates charge", "Candidates pT", 
                     "Candidates raw energy", "Candidates regressed energy", "Candidates type",
                     "N of tracksters in candidate"]

    histos_level0 = collections.defaultdict(dict)
    for release in dqm_paths.keys():
        for name in names_level0:
            histos_level0[release].update({name: dqm_files[release][name].to_hist()})

    plotter = ValidPlotter(tag, odir)
    for name in names_level0:
        plotter.plotHistos(h1=histos_level0["pre3"][name],
                           h2=histos_level0["pre4"][name],
                           label1="pre3",
                           label2="pre4",
                           modify_ticks=True,
                           savename=name)

    # Plot nested histograms for each particle type
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

    names_nested = {}
    for coll in hgcalCollections.values():
        names_nested_coll = {}

        for metric, ylabel in metrics.items():
            for step in ["energy", "pid"]:
                for axis, xlabel in axes.items():
                    names_nested_coll[f"{metric}_{coll}_{step}_{axis}"] = (ylabel, xlabel)
            # Only include "track" variables for charged particles
            if coll in ["electrons", "muons", "charged_hadrons"]:
                for axis, xlabel in axes.items():
                    names_nested_coll[f"{metric}_{coll}_track_{axis}"] = (ylabel, xlabel)
        
        names_nested[coll] = names_nested_coll

    histos_nested = collections.defaultdict(lambda: collections.defaultdict(dict))
    for release in dqm_paths.keys():
        for coll in hgcalCollections.values():
            for name in names_nested[coll]:
                histos_nested[release][coll].update({name: dqm_files[release][coll][name].to_hist()})

    for coll in hgcalCollections.values():
        makedir(os.path.join(plotter.savedir, coll))
        for name, labels in names_nested[coll].items():
            plotter.plotHistos(h1=histos_nested["pre3"][coll][name],
                               h2=histos_nested["pre4"][coll][name],
                               label1="pre3",
                               label2="pre4",
                               ylabel=labels[0],
                               xlabel=labels[1],
                               title=coll,
                               savename=coll + '/' + name)

def trackReleaseValidation(filepath1, filepath2, tag, odir):
    """
    Produces the tracking validation comparison plots.
    """
    dqm_paths = {"pre3": filepath1,
                 "pre4": filepath2}

    dqm_files = {"pre3": uproot.open(dqm_paths['pre3'])["DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTtp"],
                 "pre4": uproot.open(dqm_paths['pre4'])["DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTtp"]}

    assert dqm_paths.keys() == dqm_files.keys()

    trackCollections = {
        "GeneralTracks"           : "hltGeneral_hltAssociatorByHits", 
        "PixelTracks"             : "hltPhase2Pixel_hltAssociatorByHits", 
        # "InitialStepTracks"       : "hltInitialStepTrackSelectionHighPurity_hltAssociatorByHits", 
        # "HighPtTripletStepTracks" : "hltHighPtTripletStepTrackSelectionHighPurity_hltAssociatorByHits", 
        # "MergedSeeds"             : "hltMergedPixelHighPtTripletSeeds_hltAssociatorByHits"
    }


    names_level0 = [ "globalEfficiencies", "effic_vs_coll",
                     "fakerate_vs_coll", "pileuprate_coll",
                     "num_assoc(simToReco)_coll", "num_assoc(recoToSim)_coll" ]

    histos_level0 = collections.defaultdict(dict)
    for release in dqm_paths.keys():
        for name in names_level0:
            histos_level0[release].update({name: dqm_files[release][name].to_hist()})

    plotter = ValidPlotter(tag, odir)
    for name in names_level0:
        plotter.plotHistos(h1=histos_level0["pre3"][name],
                           h2=histos_level0["pre4"][name],
                           label1="pre3",
                           label2="pre4",
                           modify_ticks=True,
                           savename=name)

    pt_str = r"p$_{\text{T}}$"
    eta_str = r"$\eta$"
    phi_str = r"$\phi$"
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

    histos_nested = collections.defaultdict(lambda: collections.defaultdict(dict))
    for release in dqm_paths.keys():
        for coll in trackCollections.values():
            for name in names_nested:
                histos_nested[release][coll].update({name: dqm_files[release][coll][name].to_hist()})

    for coll in trackCollections.values():
        makedir(os.path.join(plotter.savedir, coll))
        for name, labels in names_nested.items():
            plotter.plotHistos(h1=histos_nested["pre3"][coll][name],
                               h2=histos_nested["pre4"][coll][name],
                               label1="pre3",
                               label2="pre4",
                               ylabel=labels[0],
                               xlabel=labels[1],
                               title=coll,
                               savename=coll + '/' + name)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make track validation plots.')
    parser.add_argument('--filepath1', type=str, required=True, help='Name of the first ROOT filepath.')
    parser.add_argument('--filepath2', type=str, required=True, help='Name of the second ROOT filepath.')
    parser.add_argument('--tag', type=str, default=None, required=True, help='Tag to uniquely identify the plots.')
    parser.add_argument('--odir', type=str, required=False, help='Path to the output directory (if not specified, save to current directory).')
    # parser.add_argument('--year', type=str, required=True,
    #                     choices=['2016', '2016APV', '2017', '2018'], help='Year')
    # parser.add_argument('--rebin', type=int, required=False, help="Rebin factor, leading to less bins.", default=1)
    # parser.add_argument('--pu', action='store_true', help='Using PU sample.')
    args = parser.parse_args()

    trackReleaseValidation(filepath1=args.filepath1, filepath2=args.filepath2, tag=args.tag, odir=args.odir)
    hgcalReleaseValidation(filepath1=args.filepath1, filepath2=args.filepath2, tag=args.tag, odir=args.odir)
