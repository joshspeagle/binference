{# Render public methods inherited from the individual-object mixin. #}
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

See :ref:`the task index and result shapes <result-task-index>` for common calls.

.. autoclass:: {{ objname }}
   :inherited-members:

   .. automethod:: __init__

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
