![pyspa banner](https://github.com/hybridlca/pyspa/blob/master/banner.png)

__pyspa__ is an object-oriented __python__ package which enables you to conduct a parametric structural path analysis on square A matrices (process or input-output) for any number of environmental, social or economic satellites and for any number of stages upstream in your supply chain (as long you have enough RAM). The package produces a SupplyChain object which includes Pathway and Node objects. Results can be exported to the csv format with a single line of code.

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







