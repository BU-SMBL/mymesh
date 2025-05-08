{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :no-members:
    :no-inherited-members:
    :show-inheritance:
    :no-special-members:

Properties
----------

{% block attributes %}
{% if attributes %}
    .. autosummary::
        :toctree:
            
    {% for item in all_attributes %}
        {%- if name == 'mesh' -%}
            {%- if not item.startswith('_') %}
            ~{{ name }}.{{ name }}.{{ item }}
            {%- endif -%}
        {%- else -%}
            {%- if not item.startswith('_') %}
            {{ name }}.{{ item }}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}
{% endif %}
{% endblock %}

Methods
-------

{% block methods %}
    .. autosummary::
        :toctree:

    {% for item in all_methods %}
        {%- if name == 'mesh' -%}
            {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
            ~{{ name }}.{{ name }}.{{ item }}
            {%- endif -%}
        {%- else -%}
            {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
                {{ name }}.{{ item }}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}
    {% for item in inherited_members %}
        {%- if name == 'mesh' -%}
            {%- if item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
            ~{{ name }}.{{ name }}.{{ item }}
            {%- endif -%}
        {%- else -%}
            {%- if item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
            ~{{ name }}.{{ name }}.{{ item }}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}
{% endblock %}