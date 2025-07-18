### PFValidation

Validation scripts for Particle Flow, as part of the NGT project (WP 3.1.1).

# Commands to produce the input files

    cmsrel CMSSW_15_1_0_pre4
    cd CMSSW_15_1_0_pre4/src
    cmsenv
    git cms-init

    cmsDriver.py step2 --step L1P2GT,HLT:NGTScouting,VALIDATION:@hltValidation \
        --conditions auto:phase2_realistic_T33 \
        --datatier GEN-SIM-DIGI-RAW,DQMIO \
        -n 10 \
        --eventcontent FEVTDEBUGHLT,DQMIO \
        --geometry ExtendedRun4D110 \
        --era Phase2C17I13M9 \
        --procModifier alpaka,ticl_v5 \
        --filein file:/eos/cms/store/relval/CMSSW_15_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_Run4D110_PU-v1/2590000//00c675dc-1517-4af7-8dd4-841e0668fefe.root \
        --fileout file:step2.root \
        --nThreads 1 \
        --process HLTX \
        --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT'

In the harvesting step, use `@hltValidation` option to compute the ratio between numerator and denominator.

    cmsDriver.py step3 \
        -s HARVESTING:@hltValidation \
        --conditions auto:phase2_realistic_T33 \
        --mc \
        --geometry ExtendedRun4D110 \
        --scenario pp \
        --filetype DQM \
        --era Phase2C17I13M9 \
        -n -1  \
        --process HLTX \
        --filein file:step2_inDQM.root \
        --fileout file:step3.root

Clone this repository:

    git clone git@github.com:elenavernazza/ValidationPlotsComparison.git

And try the plotter:

    python3 ValidationPlotsComparison/trackingValidation.py --files <path_to_file> --labels <name> --tag <your_tag> --odir <path_to_odir>
    python3 ValidationPlotsComparison/trackingValidation.py --files <path_to_file1>,<path_to_file2>,<path_to_file3> --labels <name1>,<name2>,<name3> --tag <your_tag> --odir <path_to_odir>
