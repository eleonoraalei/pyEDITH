import json, os
from typing import Union
from pyEDITH.observatory import Observatory
from pyEDITH.components import telescopes, coronagraphs, detectors


class ObservatoryBuilder:
    """
    Factory class for creating and configuring observatory objects.

    This class provides static methods to create and configure Observatory instances
    based on predefined presets or custom configurations. It manages component
    registration and creation, handles configuration validation, and maintains
    preset definitions.

    Parameters
    ----------
    PRESETS : dict
        Dictionary of predefined observatory configurations
    TOY_MODEL_COMPONENTS : dict
        Default component definitions for toy model simulations
    """

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
    def load_registry() -> dict:
        """
        Load the component registry from the registry file.

        This method reads the component registry from the JSON file and adds
        the toy model components to it. The registry contains information about
        available telescopes, coronagraphs, and detectors.

        Returns
        -------
        dict
            The component registry with toy model components added
        """

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
    def build_component_path(component_type: str, path: str) -> str:
        """
        Build the full path to a component based on its type.

        This method constructs the absolute path to a component file based on
        the component type and the provided relative path. It uses environment
        variables to determine the base directories.

        Parameters
        ----------
        component_type : str
            Type of component ('telescopes', 'detectors', or 'coronagraphs')
        path : str
            Relative path to the component file

        Returns
        -------
        str
            Absolute path to the component file

        Raises
        ------
        EnvironmentError
            If the required environment variable is not set
        ValueError
            If the component type is unknown
        """

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
    def create_observatory(config: Union[str, dict]) -> object:
        """
        Create an observatory based on the given configuration.

        This method creates an Observatory instance with telescope, coronagraph,
        and detector components as specified in the configuration. The configuration
        can be either a preset name or a custom configuration dictionary.

        Parameters
        ----------
        config : Union[str,dict]
            Either a preset name or a custom configuration dictionary

        Returns
        -------
        object
            A configured Observatory object

        Raises
        ------
        ValueError
            If the config is invalid or a component is not found
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
    def _create_component(component_type: str, keyword: str, registry: dict) -> object:
        """
        Create a component of the specified type using the keyword and registry.

        This method instantiates a component (telescope, coronagraph, or detector)
        based on the provided keyword and component registry information.

        Parameters
        ----------
        component_type : str
            Type of component ('telescopes', 'coronagraphs', or 'detectors')
        keyword : str
            Keyword identifying the specific component within its type
        registry : dict
            Component registry containing class and path information

        Returns
        -------
        object
            An instantiated component object

        Raises
        ------
        ValueError
            If the component keyword is unknown or the component class cannot be found
        """
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
    def configure_observatory(
        observatory: object, config: dict, observation: object, scene: object
    ) -> object:
        """
        Configure an existing observatory with the given configuration.

        This method applies the provided configuration parameters to an existing
        Observatory instance by calling its load_configuration method with the
        configuration and other required objects.

        Parameters
        ----------
        observatory : Observatory
            The observatory to configure
        config : dict
            Configuration parameters
        observation : Observation
            The observation object containing observational parameters
        scene : AstrophysicalScene
            The scene object containing target and environmental parameters

        Returns
        -------
        object
            The configured observatory
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
    def add_preset(cls, preset_name: str, config: dict) -> None:
        """
        Add a new preset configuration.

        This method adds a new named preset configuration to the class's PRESETS
        dictionary for future use in creating observatory instances.

        Parameters
        ----------
        preset_name : str
            Name of the new preset
        config : dict
            Configuration for the new preset

        Raises
        ------
        ValueError
            If the preset name already exists
        """

        if preset_name in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' already exists")
        cls.PRESETS[preset_name] = config

    @classmethod
    def remove_preset(cls, preset_name: str) -> None:
        """
        Remove an existing preset configuration.

        This method removes a named preset configuration from the class's PRESETS
        dictionary.

        Parameters
        ----------
        preset_name : str
            Name of the preset to remove

        Raises
        ------
        ValueError
            If the preset name does not exist
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' does not exist")
        del cls.PRESETS[preset_name]

    @classmethod
    def list_presets(cls) -> list:
        """
        List all available preset configurations.

        This method returns a list of all preset names currently defined
        in the class's PRESETS dictionary.

        Returns
        -------
        list
            Names of all available presets
        """
        return list(cls.PRESETS.keys())

    @classmethod
    def get_preset_config(cls, preset_name: str) -> dict:
        """
        Get the configuration for a specific preset.

        This method returns a copy of the configuration dictionary for the
        specified preset name, ensuring that the original preset definition
        cannot be accidentally modified.

        Parameters
        ----------
        preset_name : str
            Name of the preset

        Returns
        -------
        dict
            Configuration of the specified preset

        Raises
        ------
        ValueError
            If the preset name does not exist
        """
        preset_config = cls.PRESETS.get(preset_name)
        if preset_config is None:
            raise ValueError(f"Preset '{preset_name}' does not exist")
        return (
            preset_config.copy()
        )  # Return a copy to prevent modification of the original

    @staticmethod
    def validate_config(config: dict) -> None:
        """
        Validate a configuration dictionary.

        This method checks that a configuration dictionary contains all required
        keys and that all values are strings. It raises exceptions for any
        validation failures.

        Parameters
        ----------
        config : dict
            Configuration to validate

        Raises
        ------
        ValueError
            If the configuration is invalid
        """

        required_keys = ["telescope", "coronagraph", "detector"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        for key, value in config.items():
            if not isinstance(value, str):
                raise ValueError(f"Configuration value for {key} must be a string")

    @classmethod
    def modify_preset(cls, preset_name: str, **kwargs) -> None:
        """
        Modify a preset configuration.

        This method updates specific parameters in an existing preset configuration.
        It validates the modified configuration after changes are applied.

        Parameters
        ----------
        preset_name : str
            Name of the preset to modify
        **kwargs
            Keyword arguments representing the parameters to modify

        Raises
        ------
        ValueError
            If the preset does not exist or if invalid configuration keys are provided
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
