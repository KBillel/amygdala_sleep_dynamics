Data Format
===========

This MD files aims to described the (very) complicated data format used
in this paper in order to ease the reproduction of the figures.

CSV Files
---------

CSV files are formated in the following format :

-  Each line is a neuron.
-  The first columns allows to identify neurons

\|Rat|Day|Shank|Id|Region|Type|SessID

-  Next columns are data used in plots.

Shelves files (.dat ; .dir ; .bak)
----------------------------------

For more complicated files structures we decided to use shelves to store
the data. A shelve is a persistent python dictionary. Here is provided a
diagram showing how each shelve is organized : *Plot are often using
only the merged_session of the shelves, but individual session are kept
in case*

binned_fr_extended
~~~~~~~~~~~~~~~~~~

-  unique_sessions: *Dict* with all individual sessions

   -  Rat-08-20130708: *Dict* with output of the process_session
      generating this file

      -  session: *Dict* from bk.load.session
      -  metadata: *pd.DataFrame* metadata of the neurons of the session
      -  FR *Dict* with firing rates of neurons during
         extended_sleep/wake

         -  sleep

            -  NREM: *List*

               1) TsdFrame: firing rates of individual neurons for NREM
                  during ES1 of this session
               2) TsdFrame: firing rates of individual neurons for NREM
                  during ES2 of this session

            -  REM: same as NREM key

         -  wake

            -  WAKE_HOMECAGE: same as NREM key in sleep

      -  params: Params used for analysing this session

   -  …
   -  Rat-11-20150403

-  merged_sessions

   -  NREM

      -  FR: *List*

         1) TsdFrame of ES1
         2) TsdFrame of ES2

      -  metadata: *List*

         1) metadata of neurons corresponding to NREM/FR/ES1
         2) metadata of neurons corresponding to NREM/FR/ES2

   -  REM
   -  WAKE_HOMECAGE

transitions
~~~~~~~~~~~

-  unique_sessions: *Dict* with all individual sessions

   -  Rat-08-20130708: *Dict* with output of the process_session
      generating this file

      -  session: *Dict* from bk.load.session
      -  metadata: *pd.DataFrame* metadata of the neurons of the session
      -  transitions: *Dict* with timing of transitions of the session

         -  NREM-REM: *pd.DataFrame* with start/end of each epoch
            composing the transition
         -  REM-NREM:*pd.DataFrame* with start/end of each epoch
            composing the transition

      -  activity: *Dict* with *ndarray* of firing rates at each
         transitions (neurons,time,n_transition)

         -  NREM-REM: *ndarray* with firing rates of NREM-REM transition
            in the session
         -  REM-NREM:*ndarray* with firing rates of NREM-REM transition
            in the session

-  merged_sessions

   -  NREM-REM:*Dict*

      -  activity: *ndarray* (neurons,time) averaged firing rates across
         transitions for all neurons
      -  metadata: *pd.DataFrame* contains the metadata of each neurons

   -  REM-NREM

network_metrics
~~~~~~~~~~~~~~~

-  unique_sessions: *Dict* with all individual sessions

   -  Rat-08-20130708: *Dict* with output of the process_session
      generating this file

      -  session: *Dict* from bk.load.session
      -  metrics: *Dict*

         -  raw

            -  eib: *Tuple[ArrayLike,ArrayLike]* -> t,values
            -  cv: *Tuple[ArrayLike,ArrayLike]* -> t,values
            -  sync: *Tuple[ArrayLike,ArrayLike]* -> t,values

         -  averaged: *Dict* with different kind of averaged

            -  epochs: *pd.DataFrame* start,end,state,metrics
            -  thirds *pd.DataFrame* start,end,state,metrics
            -  nbins *pd.DataFrame* start,end,state,metrics

      -  params: *Dict*

-  merged_sessions

   -  epochs: *Dict[str,Dict]*

      -  metric: *Dict[str,ArraylLike]*-> 1 values per epochs

   -  thirds: *Dict[str,ArrayLike]*

      -  metric: *Dict[str,ArraylLike]*-> n_epochs x 3

   -  nbins: *Dict[str,ArrayLike]*

      -  metric: *Dict[str,ArraylLike]*-> n_epochs x nbins

oscillations
~~~~~~~~~~~~

-  unique_sessions: *Dict* with all individual sessions

   -  Rat-08-20130708: *Dict* with output of the process_session
      generating this file

      -  session: *Dict* from bk.load.sessdion
      -  power: *Dict*

         -  raw: *Dict*

            -  left: *Dict*

               -  delta: *Tuple[ArrayLike,ArrayLike]* -> t,values
               -  theta: *Tuple[ArrayLike,ArrayLike]* -> t,values
               -  gammas: *Tuple[ArrayLike,ArrayLike]* -> t,values

         -  averaged: *Dict*

            -  left: *pd.DataFrame* start,end,state,bands

      -  params: *Dict*

-  merged_session: *Dict*

   -  left: *Dict*

      -  delta: *Dict[str,ArrayLike]* -> n_epochs x nbins
      -  theta: *Dict[str,ArrayLike]* -> n_epochs x nbins

   -  right

      -  delta: *Dict[str,ArrayLike]* -> n_epochs x nbins
      -  theta: *Dict[str,ArrayLike]* -> n_epochs x nbins
