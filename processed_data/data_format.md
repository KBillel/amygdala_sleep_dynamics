# Data Format

This MD files aims to described the (very) complicated data format used in this paper in order to ease the reproduction of the figures.

## CSV Files

CSV files are formated in the following format :

- Each line is a neuron.
- The first columns allows to identify neurons

|Rat|Day|Shank|Id|Region|Type|SessID

- Next columns are data used in plots.

## Shelves files (.dat ; .dir ; .bak)

For more complicated files structures we decided to use python shelves to store the data.

A shelve is a persistent dictionary.
Here is provided a diagram showing how each shelve is organized : 

### binned_fr_extended

binned_fr_extended:

- unique_sessions
  - Rat-08-20130708
    - session -> from bk.load.session
    - FR
      - sleep
        - NREM (List)
          1) TsdFrame: firing rates of individual neurons for NREM during ES1 of this session
          2) TsdFrame: firing rates of individual neurons for NREM during ES2 of this session
        - REM same as NREM key
      - wake
        - WAKE_HOMECAGE - same as NREM key in sleep
    - metadata -> metadata of the neurons of the session
    - params -> Params used for analysing this session
  - ...
  - Rat-11-20150403
- merged_sessions
  - NREM
    - FR (List)
      1) TsdFrame of ES1
      2) TsdFrame of ES2
    - metadata (List)
      1) metadata of neurons corresponding to NREM/FR/ES1  
      2) metadata of neurons corresponding to NREM/FR/ES2
  - REM
  - WAKE_HOMECAGE
