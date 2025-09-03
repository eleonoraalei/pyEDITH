# Advanced Options

## Change a preset

`pyEDITH` has some standard presets for the currently assumed Exploratory Analytic Cases for HWO. If you need to specify a different preset (e.g. changing the coronagraph), you can replace this parameter keyword that we use in the [Imaging Tutorial](imaging_tutorial.ipynb) and the [Spectroscopy Tutorial](spectroscopy_tutorial.ipynb):


```python
parameters["observatory_preset"] = "EAC1"
```

with three keywords:

```python
parameters["telescope_type"] = "EAC1"
parameters["coronagraph_type"] = "MyNewCoronagraph"
parameters["detector_type"] = "EAC1"
```


## Add your own YIP coronagraph

1. Add your new Coronagraph YIP folder to the YIP folder (see [Installation](installation.md)). For example, we assume this folder is called `NewCoronagraph`.
2. Open `src/components/registry.json`. This file contains the dictionary that connects your favorite keyword to the name of the coronagraph YIP folder. It will look something like this:

```json
{
    "telescopes": {
        "EAC1": {
            "class": "EACTelescope",
            "path": ""
        }
    },
    "coronagraphs": {
        "LUVOIR": {
            "class": "CoronagraphYIP",
            "path": "usort_offaxis_ovc"
        },
    },
    "detectors": {
        "EAC1": {
            "class": "EACDetector",
            "path": ""
        }
    }
}
```

3. Add to the registry the keyword and the dictionary establishing the class of the coronagraph (for all the YIP-like coronagraphs, it will be `CoronagraphYIP`) and then the name of the folder.

```json
{
    "telescopes": {
        "EAC1": {
            "class": "EACTelescope",
            "path": ""
        }
    },
    "coronagraphs": {
        "LUVOIR": {
            "class": "CoronagraphYIP",
            "path": "usort_offaxis_ovc"
        },
        "MyNewCoronagraph": {
            "class": "CoronagraphYIP",
            "path": "NewCoronagraph"
        },
    },
    "detectors": {
        "EAC1": {
            "class": "EACDetector",
            "path": ""
        }
    }
}
```

4. If you want to use this coronagraph, you should change the observatory preset in your parameters (see [above](advanced.md#change-a-preset)).