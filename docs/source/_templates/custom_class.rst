{{ fullname | escape | underline}}

.. autoclass:: {{ fullname }}

   {% block class_docstring %}
   {{ obj.__doc__.split('\n')[0] }}

   {% set lines = obj.__doc__.split('\n')[1:] %}
   {% set in_attributes = false %}
   {% for line in lines %}
   {% if line.strip().startswith('Attributes') %}
   {% set in_attributes = true %}

   Attributes:
   {% elif in_attributes and line.strip() and not line.startswith('---') %}
   {% set parts = line.split(':') %}
   {% if parts|length > 1 %}
   * {{ parts[0].strip() }} ({{ parts[1].strip() }}) - {{ lines[loop.index].strip() }}
   {% endif %}
   {% elif in_attributes and not line.strip() %}
   {% set in_attributes = false %}
   {% endif %}
   {% endfor %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}