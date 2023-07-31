Generating figure 2: BLA is highly active during REM sleep
==========================================================


Overview
--------

To generate figure 2 one needs to:

1. Execute processing/fr_states.py : 

.. code-block:: bash
   :linenos:

   python processing/fr_states.py

This will load, session by session, the data set and compute firing rates of all the neurons with various conditions.

2. Execute plots/plot_fr.py

.. code-block:: bash
   :linenos:

   python plots/plot_fr.py

This will generate an svg in files in plots/figures. 


Details
--------

fr_states.py calls :py:func:`~processing.fr_states.process_all_sessions` with following parameters :

* base_folder : Folder of the dataset.
* params : a dict that contains parameter specific to extended wake and extended sleep. 
   * State : compute extended period of 'wake' or 'sleep'
   * sleep_th : minimal or maximum amount of time in sleep (minutes)
   * wake_th : minimal or maximal amount of time in wake (minutes)
   * sub_states : Compute the firing rates for separate substates (NREM/REM for instance)
* save : a boolean in order to save every session to a shelve.



:py:func:`~processing.fr_states.process_all_sessions` calls :py:func:`~processing.fr_states.process_session`.

:py:func:`~processing.fr_states.process_session` proced to save each session :

* In a shelves located at processed_data/binned_fr_extended with a json files with the parameters
* In CSV files :
   * delta_extended.csv/json 
   * rem_on.csv/json
   * states_fr.csv/json
 
:py:func:`~processing.fr_states.process_all_sessions` will also save merged processed data after running :py:func:`~processing.fr_states.merge_extended`

Once processing done, :py:mod:`~plots.plot_fr` will generates the figure.
Variable quantile_state, will define if neurons are sorted base on firing rates during WAKE or SLEEP.

Panel table
-----------


.. list-table::
   :header-rows: 1

   * - figure
     - panel
     - function
     - parameters
   * - 2
     - A
     - :py:func:`~plots.plot_fr.boxenplot_firing_rates`
     - df,"BLA",axes
   * - 2
     - B
     - :py:func:`~plots.plot_fr.cumsum_curves_firing_rates`
     - df,"BLA",['NREM','REM','WAKE_HOMECAGE'],axes
   * - 2
     - B
     - :py:func:`~plots.plot_fr.cumsum_curves_firing_rates`
     - df,"BLA",['NREM','REM','WAKE_HOMECAGE'],axes
   * - 2
     - C
     - :py:func:`~plots.plot_fr.plot_histograms_firing_rates`
     - df,"BLA",quantile_state,axes
   * - 2
     - D-top
     - :py:func:`~plots.plot_fr.plot_transitions_panel`
     - transitions,df,stru,None,None,params,NREM-REM,axes
   * - 2
     - D-bot
     - :py:func:`~plots.plot_fr.plot_transitions_panel`
     - transitions,df,stru,zscore,quantile_state,params,NREM-REM,axes
   * - 2
     - E
     - :py:func:`~plots.plot_fr.proportion_rem_on`
     - rem_on_off,"BLA",axes
   * - 2
     - F
     - :py:func:`~plots.plot_fr.corr_rem_nrem_fr`
     - df,"BLA","WAKE_HOMECAGE",axes
   * - 2
     - G
     - :py:func:`~plots.plot_fr.corr_rem_nrem_fr`
     - df,"BLA","WAKE_HOMECAGE",axes
