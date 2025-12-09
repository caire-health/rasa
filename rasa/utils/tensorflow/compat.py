"""Compatibility layer for TensorFlow 2.16+ and Keras 3.0+.

This module provides compatibility shims for APIs that changed between
TensorFlow 2.12/Keras 2.12 and TensorFlow 2.16+/Keras 3.0+.
"""
import tensorflow as tf

# Import Model - try keras 3.0+ first, then fallback to tensorflow.keras
try:
    from keras import Model
except ImportError:
    from tensorflow.keras import Model

# Import tf_utils - try keras 3.0+ first, then tensorflow.python.keras.utils
try:
    from keras.utils import tf_utils
except ImportError:
    try:
        from tensorflow.python.keras.utils import tf_utils
    except ImportError:
        # If tf_utils is not available, create a compatibility function
        def sync_to_numpy_or_python_type(outputs):
            """Convert TensorFlow tensors to numpy arrays or Python types."""
            if isinstance(outputs, dict):
                return {k: sync_to_numpy_or_python_type(v) for k, v in outputs.items()}
            elif isinstance(outputs, (list, tuple)):
                return type(outputs)(sync_to_numpy_or_python_type(item) for item in outputs)
            elif tf.is_tensor(outputs):
                return outputs.numpy() if hasattr(outputs, 'numpy') else outputs
            else:
                return outputs
        
        class tf_utils:
            sync_to_numpy_or_python_type = staticmethod(sync_to_numpy_or_python_type)

# Import smart_cond - try keras 3.0+ first, then tensorflow.python.keras.utils
try:
    from keras.utils.control_flow_util import smart_cond
except ImportError:
    try:
        from tensorflow.python.keras.utils.control_flow_util import smart_cond
    except ImportError:
        # In Keras 3.0+, smart_cond may not be available, use tf.cond as fallback
        def smart_cond(condition, true_fn, false_fn):
            """Compatibility wrapper for smart_cond using tf.cond."""
            return tf.cond(condition, true_fn, false_fn)

__all__ = ['Model', 'tf_utils', 'smart_cond']

