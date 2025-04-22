import json, os

from pyEDITH.observatory import Observatory
from pyEDITH.components import telescopes, coronagraphs, detectors


class ObservatoryBuilder:

    # Default presents for the currently implemented concepts
    PRESETS = {
        "ToyModel": {
            "telescope": "ToyModel",
            "coronagraph": "ToyModel",
            "detector": "ToyModel",
        },
        "EAC1": {
            "telescope": "EAC1",
            "coronagraph": "LUVOIR",
            "detector": "EAC1",
        },
        "EAC2": {
            "telescope": "EAC2",
            "coronagraph": "LUVOIR",
            "detector": "EAC2",
        },
        "EAC3": {
            "telescope": "EAC3",
            "coronagraph": "LUVOIR",
            "detector": "EAC3",
        },
    }

    # Hardcoded registry for ToyModel since we do not expect it to be used much
    TOY_MODEL_COMPONENTS = {
        "telescopes": {"class": "ToyModelTelescope", "path": None},
        "coronagraphs": {"class": "ToyModelCoronagraph", "path": None},
        "detectors": {"class": "ToyModelDetector", "path": None},
    }

    @staticmethod
    def load_registry():
        registry_path = os.path.join(
            os.path.dirname(__file__), "components/registry.json"
        )
        with open(registry_path, "r") as f:
            registry = json.load(f)

        # Add ToyModel components to the registry
        for (
            component_type,
            toy_component,
        ) in ObservatoryBuilder.TOY_MODEL_COMPONENTS.items():
            registry[component_type]["ToyModel"] = toy_component
        return registry

    @staticmethod
    def build_component_path(component_type, path):
        if component_type == "telescopes" or component_type == "detectors":
            base_dir = os.environ.get("SCI_ENG_DIR")
            if not base_dir:
                raise EnvironmentError("SCI_ENG_DIR environment variable not set")
        elif component_type == "coronagraphs":
            base_dir = os.environ.get("YIP_CORO_DIR")
            if not base_dir:
                raise EnvironmentError("YIP_CORO_DIR environment variable not set")
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        return os.path.join(base_dir, path)

    @staticmethod
    def create_observatory(config):
        """
        Create an observatory based on the given configuration.

        Parameters:
        config (str or dict): Either a preset name or a custom configuration dictionary.

        Returns:
        Observatory: A configured Observatory object.

        Raises:
        ValueError: If the config is invalid or a component is not found.
        """

        registry = ObservatoryBuilder.load_registry()

        # Check that the config is either a string with a valid keyword...
        if isinstance(config, str):
            if config not in ObservatoryBuilder.PRESETS:
                raise ValueError(f"Unknown preset observatory: {config}")
            preset_config = ObservatoryBuilder.PRESETS[config]
            telescope = ObservatoryBuilder._create_component(
                "telescopes", preset_config["telescope"], registry
            )

            coronagraph = ObservatoryBuilder._create_component(
                "coronagraphs", preset_config["coronagraph"], registry
            )

            detector = ObservatoryBuilder._create_component(
                "detectors", preset_config["detector"], registry
            )

        elif isinstance(config, dict):
            telescope = ObservatoryBuilder._create_component(
                "telescopes", config["telescope"], registry
            )
            coronagraph = ObservatoryBuilder._create_component(
                "coronagraphs", config["coronagraph"], registry
            )
            detector = ObservatoryBuilder._create_component(
                "detectors", config["detector"], registry
            )
        else:
            raise ValueError("Invalid configuration type")

        observatory = Observatory()
        observatory.telescope = telescope
        observatory.coronagraph = coronagraph
        observatory.detector = detector

        return observatory

    @staticmethod
    def _create_component(component_type, keyword, registry):
        component_info = registry[component_type].get(keyword)
        if not component_info:
            raise ValueError(f"Unknown {component_type} keyword: {keyword}")
        if component_info["path"] is not None:
            path = ObservatoryBuilder.build_component_path(
                component_type, component_info["path"]
            )
        else:
            path = None
        module = globals()[component_type]

        # Initalize the component (by specifying the path)
        try:
            component_class = getattr(module, component_info["class"])
            return component_class(path, keyword)
        except:
            raise ValueError(f"Unknown component class: {component_info['class']}")

    @staticmethod
    def configure_observatory(observatory, config, observation, scene):
        """
        Configure an existing observatory with the given configuration.

        Parameters:
        observatory (Observatory): The observatory to configure.
        config (dict): Configuration parameters.

        Returns:
        Observatory: The configured observatory.
        """
        observatory.load_configuration(config, observation, scene)
        return observatory

    # EXTRA: we can use this to have the user add more components from command line
    #  @staticmethod
    # def add_component(component_type, keyword, class_name, path=None):
    #     if keyword.lower() == "toymodel":
    #         raise ValueError("Cannot modify ToyModel components")

    #     registry = ObservatoryBuilder.load_registry()
    #     registry[component_type][keyword] = {"class": class_name, "path": path}
    #     registry_path = os.path.join(os.path.dirname(__file__), 'registry.json')
    #     with open(registry_path, 'w') as f:
    #         json.dump(registry, f, indent=4)
    #     print(f"Added {component_type} '{keyword}' to the registry.")

    @classmethod
    def add_preset(cls, preset_name, config):
        """
        Add a new preset configuration.

        Parameters:
        preset_name (str): Name of the new preset.
        config (dict): Configuration for the new preset.

        Raises:
        ValueError: If the preset name already exists.
        """
        if preset_name in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' already exists")
        cls.PRESETS[preset_name] = config

    @classmethod
    def remove_preset(cls, preset_name):
        """
        Remove an existing preset configuration.

        Parameters:
        preset_name (str): Name of the preset to remove.

        Raises:
        ValueError: If the preset name does not exist.
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' does not exist")
        del cls.PRESETS[preset_name]

    @classmethod
    def list_presets(cls):
        """
        List all available preset configurations.

        Returns:
        list: Names of all available presets.
        """
        return list(cls.PRESETS.keys())

    @classmethod
    def get_preset_config(cls, preset_name):
        """
        Get the configuration for a specific preset.

        Parameters:
        preset_name (str): Name of the preset.

        Returns:
        dict: Configuration of the specified preset.

        Raises:
        ValueError: If the preset name does not exist.
        """
        preset_config = cls.PRESETS.get(preset_name)
        if preset_config is None:
            raise ValueError(f"Preset '{preset_name}' does not exist")
        return (
            preset_config.copy()
        )  # Return a copy to prevent modification of the original

    @staticmethod
    def validate_config(config):
        """
        Validate a configuration dictionary.

        Parameters:
        config (dict): Configuration to validate.

        Raises:
        ValueError: If the configuration is invalid.
        """
        required_keys = ["telescope", "coronagraph", "detector"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        for key, value in config.items():
            if not isinstance(value, str):
                raise ValueError(f"Configuration value for {key} must be a string")

    @classmethod
    def modify_preset(cls, preset_name, **kwargs):
        """
        Modify an existing preset configuration.

        Parameters:
        preset_name (str): Name of the preset to modify.
        **kwargs: Key-value pairs to update in the preset configuration.

        Raises:
        ValueError: If the preset name does not exist or if invalid keys are provided.
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' does not exist")

        preset_config = cls.PRESETS[preset_name]
        valid_keys = set(preset_config.keys())

        for key, value in kwargs.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid configuration key: {key}")
            preset_config[key] = value

        cls.validate_config(preset_config)
