![pyspa banner](https://github.com/hybridlca/pyspa/blob/master/banner.png)

__pyspa__ is an object-oriented __python__ package which enables you to conduct a parametric structural path analysis on square A matrices (process or input-output) for any number of environmental, social or economic satellites/flows and for any number of stages upstream in your supply chain (as long you have enough RAM). The package produces a SupplyChain object which includes Pathway and Node objects (among others). Results can be exported to the csv format with a single line of code.

The concept behind __pyspa__ was driven by the lack of open source code to conduct structural path analysis in a robust and object-oriented manner.

## Getting Started

### Prerequisites

You will need __python__ to run this package as well as the following python packages:
1. [numpy](https://www.numpy.org/)
2. [pandas](https://pandas.pydata.org/)

### Installing
Download and install the package from pip

```
pip install pyspa
```

## Testing pyspa

Identify the template files in the installed directory, or download them directly from the [github repository](https://github.com/hybridlca/pyspa/). The template files include:

1. A_matrix_template.csv
2. Infosheet_template.csv
3. Thresholds_template.csv

Once you have located these files, __you need to run a single function__ that will read the data, conduct the structural path analysis and return a SupplyChain object, as per the following code.

```
sc = pyspa.get_spa(target_id = 70, max_stage = 10, a_matrix_file_path ='A_matrix_template.csv', infosheet_file_path='Infosheet_template.csv', thresholds_file_path='Thresholds_template.csv')
```

This will return your SupplyChain object which has numerous methods. Read the [documentation](http://htmlpreview.github.io/?https://github.com/hybridlca/pyspa/blob/master/pyspa_v1.0_documentation.html) for more information.

To export the structural path analysis to a __csv__ file, use the built-in method.

```
sc.export_to_csv('spa_results.csv')
```

To __save__ your SupplyChain object and avoid having to recalculate everything (this uses pickle):

```
sc.save('supply_chain.sc')
```

To __load__ a previously saved SupplyChain object:

```
loaded_sc = pyspa.load_instance_from_file('supply_chain.sc', pyspa.SupplyChain)
```
The __detailed documentation__ is available [here](http://htmlpreview.github.io/?https://github.com/hybridlca/pyspa/blob/master/pyspa_v1.0_documentation.html)

## Input files

### Description
The package requires three csv files to be able to conduct a structural path analysis:
1. A __square__ technological matrix, aka an A matrix
2. An infosheet listing all sectors or processes, along with the direct and total intensities/multipliers/requirements for any number of environmental/economic/social satellites, and their metadata
3. The cut-off thresholds used to trim the supply chain branches for each satellite.

These csv files must be formatted in a certain way for the code to work. The formatting requirements are described below.

### Formatting

#### Square technological matrix (A matrix)
The A matrix should be provided in a single csv file, regardless of its size (we have tried the code on 15k×15k matrix so far, and it works fine). It must be formatted as follows:

+ The top row must be the indexes of the sectors/processes, numbered from 1 to n.
+ The rest of the matrix comes underneath that row.
+ No text headers nor text content

| 1        | ...           | n  |
| :-------------: |:-------------:| :-----:|
| <A matrix: input from 1 into 1> | <A matrix: input from 1 into ...> | <A matrix: input from 1 into n> |
| <A matrix: input from ... into 1> | <A matrix: input from ... into ...> | <A matrix: input from ... into n> |
| <A matrix: input from n into 1>  | <A matrix: input from n into ...> | <A matrix: input from n into n> |
  
#### Infosheet

The infosheet must contain mandatory columns and at least one environmental/social/economic satellite/flow. It must be formatted as follows (all headers are case sensitive):

+ The __first column__ has a header called __"Sector ID"__ and contains the IDs of each sector/process from 1 to n. These IDs match those included as a header in the __A matrix__.
+ The __second column__ has a header called __"Name"__ and contains the name of each sector/process. It is highly recommended to have unique names as the csv output of the package uses names (not IDs).
+ The __third column__ has a header called __"Unit"__ and contains the functional unit of each sector/process. It is usually a financial currency for input-output sectors (e.g. AUD, USD, EUR, YEN, etc.) and can be a physical unit for processes (e.g. kg, m³, tkm, etc.).
+ The __fourth column__ has a header called __"Region"__ and contains the region of each sector processs. If you are not working with multiregional data, simply populate this column with the name of the region for your data (for instance in the template file, the region for all sectors is _"Australia"_.
+ From the __fifth column__ onwards you need to include at least one satellite/flow. Satellites/flows are included using __two columns__:
  +The first column contains the direct intensity/multiplier/requirement for your satellite/flow and has a header in the following format: 
  __DR\_<satellite/flow_name>\_(<satellite/flow_unit>)__
  For example, for greenhouse gas emissions, you can write: __DR\_GHGe\_(kgCO<sub>2</sub>e)__
  +The second column contains the total intensity/multiplier/requirement for your satellite/flow and has a header in the following format: 
  __TR\_<satellite/flow_name>\_(<satellite/flow_unit>)__
  For example, for greenhouse gas emissions, you can write: __TR\_GHGe\_(kgCO<sub>2</sub>e)__
  
You can add as many satellites as you need to the infosheet. The code will detect them automatically, as long as their headers are formatted as above.
You can also add any other metadata column for your sectors/processes, and then access them through manual coding using the predefined method on your _Node_ objects: _get\_node\_attribute_. See the [__detailed documentation__](http://htmlpreview.github.io/?https://github.com/hybridlca/pyspa/blob/master/pyspa_v1.0_documentation.html) for more details.

#### Thresholds

The thresholds csv is by far the simplest csv file to provide. It contains only __two columns__ and must be formatted as below:

+ The __first column__ has a header called __"Flow"__ which contains the name of each satellite/flow that you are using, e.g. GHGe. The name of the satellite/flow must be exactly the same as what is contained in the DR and TR headers of the infosheet, but without the __DR/TR\___ prefix and without the __\_(<satellite/flow_unit>)__ suffix.
+ The __second column__ has a header called __"Value"__ which contains the threshold value of each satellite/flow that you are using, e.g. GHGe. This value is usually very low. For common environmental satellites/flows, such as water(kL), energy(GJ) and greenhouse gas emissions(kgCO<sub>2<\sub>e), we use threshold values for input-output data in the range of 0.000 1 and 0.000 000 000 1. The lower the threshold, the more supply chain nodes you consider, the longer the structural path analysis will take.
 
## CSV output file

The csv output file contains some metadata on the structural path analysis itself and then lists, for each satellite/flow, the pathways extracted, by order of significance in terms of the direct intensity/multiplier/requirement of the last node in that pathway. The columns for these listing are:
+ The percentage of contribution of that last node in that pathway, to the total intensity/multiplier/requirement of the selected sector/process is provided
+ The value of the corresponding direct intensity/multiplier/requirement
+ The value of the corresponding total intensity/multiplier/requirement
+ The name of each node in the pathway, for each stage of the supply chain (1 to n).

The direct intensity/multiplier/requirement of the selected sector/process is referred to as _DIRECT (Stage 0)_. _Stage 1_ refers to the first stage upstream in the supply chain, _Stage 2_ the following stage, all the way to _Stage m_ as selected at the start. We recommend using around __10 stages__ upstream for process data, and __8 stages__ upstream for input-output data, based on our experience. But these values might differ.

## Built with:

+ [pycharm](https://www.jetbrains.com/pycharm/)
+ [sourcetree](https://www.sourcetreeapp.com/)
+ Sweat, tears, Belgian beers, and coffee from Castro's

## Authors and contributors

### Authors
+ [André Stephan](https://github.com/andrestephan1) - _overall design, implementation, testing and debugging_ - [ORCID](https://orcid.org/0000-0001-9538-3830)
+ [Paul-Antoine Bontinck](https://github.com/pa-bontinck) - _optimisation, implementation, testing and debugging_ - [ORCID](https://orcid.org/0000-0002-4072-1334)
### Contributors
+ [Robert H Crawford](https://github.com/rhcr) - _project leader and theoretical guidance_ - [ORCID](https://orcid.org/0000-0002-0189-3221)

## License
This project is shared under a GNU General Public License v3.0. See the [LICENSE](../master/LICENSE) file for more information.

## Acknowledgments

This project was funded by the __Australian Research Council Discovery Project DP150100962__ at the [University of Melbourne](https://unimelb.edu.au/), Australia. As such, we are endebted to Australian taxpayers for making this work possible and to the University of Melbourne for providing the facilities and intellectual space to conduct this research. The code for the base method for conducting the structural path analysis is inspired from the code of late __[A/Prof Graham Treloar](https://new.gbca.org.au/news/gbca-news/how-legacy-late-green-building-researcher-lives/)__ at the University of Melbourne, who pioneered a Visual Basic Script in his [PhD thesis](https://dro.deakin.edu.au/eserv/DU:30023444/treloar-comprehensiveembodied-1998.pdf) to conduct a structural path analysis in 1997.







