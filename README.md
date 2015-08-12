Grow Hubway: identifying the best locations for new Hubway stations
===================================================================


## Motivation

Hubway is Boston's bike-sharing system.  It started operations in July 2011 and
the number of people taking rides every day has been growing year over year
since its inception.  Now is an excellent time for Hubway to think carefully
about growing their system of bike stations.  The purpose of this software is
to provide a quantitative prediction of the best locations in the greater
Boston area for new Hubway stations.

## Overview of workflow

views.py: top-level routine that runs in the background on the AWS EC2
instance.  Generates web pages and their content dynamically using Flask.
Calls gridpredict to do the heavy lifting.
