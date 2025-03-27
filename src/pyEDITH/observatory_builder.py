from pyEDITH.observatory import Observatory
from pyEDITH.components import telescopes, coronagraphs, detectors


class ObservatoryBuilder:

    # Default presents for the currently implemented concepts
    PRESETS = {
        "ToyModel": {
            "telescope": "ToyModelTelescope",
            "coronagraph": "ToyModelCoronagraph",
            "detector": "ToyModelDetector",
        },
        "EAC1": {
            "telescope": "EAC1Telescope",
            "coronagraph": "CoronagraphYIP",
            "detector": "EAC1Detector",
        },
    }

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

        # Check that the config is either a string with a valid keyword...
        if isinstance(config, str):
            config = ObservatoryBuilder.PRESETS.get(config)
            if config is None:
                raise ValueError(f"Unknown preset observatory: {config}")
        # Or a dictionary that sets up the configuration
        elif not isinstance(config, dict):
            raise ValueError(
                "Config must be either a string (preset name) or a dictionary (custom configuration)"
            )

        # Initialize the components, or pick the ToyModel versions if not available.
        
        telescope = ObservatoryBuilder._create_component(
            config.get("telescope", "ToyModelTelescope"), telescopes
        )  
        coronagraph = ObservatoryBuilder._create_component(
            config.get("coronagraph", "ToyModelCoronagraph"), coronagraphs
        )
        detector = ObservatoryBuilder._create_component(
            config.get("detector", "ToyModelDetector"), detectors
        )

        observatory = Observatory()
        observatory.telescope = telescope
        observatory.coronagraph = coronagraph
        observatory.detector = detector

        return observatory

    @staticmethod
    def _create_component(component_name, module):
        """
        Create a component instance based on its name and module.

        Parameters:
        component_name (str): Name of the component class.
        module: Module containing the component class.

        Returns:
        object: An instance of the specified component.

        Raises:
        ValueError: If the component is not found in the module.
        """
        # Search for the class that has the required name in the desired module (like doing module.component_name)
        # If it finds that, creates instance, otherwise fails.
        try:
            component_class = getattr(module, component_name)
            return component_class()
        except AttributeError:
            raise ValueError(f"Unknown component: {component_name}")

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
        if preset_name.lower() in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' already exists")
        cls.PRESETS[preset_name.lower()] = config

    @classmethod
    def remove_preset(cls, preset_name):
        """
        Remove an existing preset configuration.

        Parameters:
        preset_name (str): Name of the preset to remove.

        Raises:
        ValueError: If the preset name does not exist.
        """
        if preset_name.lower() not in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' does not exist")
        del cls.PRESETS[preset_name.lower()]

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
        preset_config = cls.PRESETS.get(preset_name.lower())
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
        if preset_name.lower() not in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' does not exist")

        preset_config = cls.PRESETS[preset_name.lower()]
        valid_keys = set(preset_config.keys())

        for key, value in kwargs.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid configuration key: {key}")
            preset_config[key] = value

        cls.validate_config(preset_config)
