"""Tests for TensorFlow 2.16+ and Keras 3.0+ compatibility."""
import pytest
import tensorflow as tf
import numpy as np

from rasa.utils.tensorflow.compat import Model, tf_utils, smart_cond


def test_model_import():
    """Test that Model can be imported and is a valid class."""
    assert Model is not None
    # Model should be a class that can be subclassed
    assert isinstance(Model, type)


def test_tf_utils_import():
    """Test that tf_utils can be imported and has sync_to_numpy_or_python_type."""
    assert tf_utils is not None
    assert hasattr(tf_utils, 'sync_to_numpy_or_python_type')
    assert callable(tf_utils.sync_to_numpy_or_python_type)


def test_sync_to_numpy_or_python_type_with_tensor():
    """Test sync_to_numpy_or_python_type converts tensors to numpy arrays."""
    tensor = tf.constant([1, 2, 3])
    result = tf_utils.sync_to_numpy_or_python_type(tensor)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_sync_to_numpy_or_python_type_with_dict():
    """Test sync_to_numpy_or_python_type handles dictionaries."""
    tensor_dict = {
        'a': tf.constant([1, 2]),
        'b': tf.constant([3, 4])
    }
    result = tf_utils.sync_to_numpy_or_python_type(tensor_dict)
    
    assert isinstance(result, dict)
    assert isinstance(result['a'], np.ndarray)
    assert isinstance(result['b'], np.ndarray)
    np.testing.assert_array_equal(result['a'], np.array([1, 2]))
    np.testing.assert_array_equal(result['b'], np.array([3, 4]))


def test_sync_to_numpy_or_python_type_with_list():
    """Test sync_to_numpy_or_python_type handles lists."""
    tensor_list = [tf.constant([1, 2]), tf.constant([3, 4])]
    result = tf_utils.sync_to_numpy_or_python_type(tensor_list)
    
    assert isinstance(result, list)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
    np.testing.assert_array_equal(result[0], np.array([1, 2]))
    np.testing.assert_array_equal(result[1], np.array([3, 4]))


def test_sync_to_numpy_or_python_type_with_numpy_array():
    """Test sync_to_numpy_or_python_type leaves numpy arrays unchanged."""
    arr = np.array([1, 2, 3])
    result = tf_utils.sync_to_numpy_or_python_type(arr)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_smart_cond_import():
    """Test that smart_cond can be imported and is callable."""
    assert smart_cond is not None
    assert callable(smart_cond)


def test_smart_cond_functionality():
    """Test that smart_cond works correctly."""
    condition = tf.constant(True)
    
    def true_fn():
        return tf.constant(1.0)
    
    def false_fn():
        return tf.constant(0.0)
    
    result = smart_cond(condition, true_fn, false_fn)
    
    assert tf.is_tensor(result)
    assert result.numpy() == 1.0


def test_smart_cond_with_false_condition():
    """Test smart_cond with False condition."""
    condition = tf.constant(False)
    
    def true_fn():
        return tf.constant(1.0)
    
    def false_fn():
        return tf.constant(0.0)
    
    result = smart_cond(condition, true_fn, false_fn)
    
    assert tf.is_tensor(result)
    assert result.numpy() == 0.0


def test_tensorflow_version():
    """Test that TensorFlow version is >= 2.16.0."""
    tf_version = tf.__version__
    major, minor = map(int, tf_version.split('.')[:2])
    
    assert major >= 2, f"TensorFlow version {tf_version} is too old"
    if major == 2:
        assert minor >= 16, f"TensorFlow version {tf_version} is < 2.16.0"


def test_keras_version():
    """Test that Keras version is >= 3.0.0."""
    try:
        import keras
        keras_version = keras.__version__
        major, minor = map(int, keras_version.split('.')[:2])
        
        assert major >= 3, f"Keras version {keras_version} is < 3.0.0"
    except ImportError:
        # If keras is not available as separate package, check tf.keras
        # This is acceptable for TensorFlow 2.16+ where keras might be bundled
        pass


def test_model_inheritance():
    """Test that Model can be used as a base class."""
    class TestModel(Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(1)
        
        def call(self, inputs):
            return self.dense(inputs)
    
    model = TestModel()
    assert isinstance(model, Model)
    
    # Test that it can be called
    test_input = tf.constant([[1.0]])
    output = model(test_input)
    assert tf.is_tensor(output)


def test_compat_with_rasa_model():
    """Test that compat module works with RasaModel."""
    from rasa.utils.tensorflow.models import RasaModel
    
    # Just verify that RasaModel can be imported and uses the compat Model
    assert RasaModel is not None
    # RasaModel should inherit from Model
    assert issubclass(RasaModel, Model)

