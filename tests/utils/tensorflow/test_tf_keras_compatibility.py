"""Integration tests for TensorFlow 2.16+ and Keras 3.0+ compatibility.

These tests verify that the codebase works correctly with the updated
TensorFlow and Keras versions, ensuring no breaking changes were introduced.
"""
import pytest
import tensorflow as tf
import numpy as np

from rasa.utils.tensorflow.transformer import MultiHeadAttention, TransformerEncoder
from rasa.utils.tensorflow.layers import SparseDropout
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.constants import LABEL, IDS, SENTENCE
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE


def test_multi_head_attention_with_smart_cond():
    """Test that MultiHeadAttention works with the new smart_cond implementation."""
    mha = MultiHeadAttention(units=64, num_heads=4)
    
    # Create dummy inputs
    query = tf.random.normal((2, 10, 64))
    key = tf.random.normal((2, 10, 64))
    value = tf.random.normal((2, 10, 64))
    
    # Test with training=True (uses smart_cond)
    output = mha(query, key, value, training=True)
    assert output.shape == (2, 10, 64)
    
    # Test with training=False (uses smart_cond)
    output = mha(query, key, value, training=False)
    assert output.shape == (2, 10, 64)


def test_sparse_dropout_with_smart_cond():
    """Test that SparseDropout works with the new smart_cond implementation."""
    dropout = SparseDropout(rate=0.5)
    
    # Create a sparse tensor
    indices = tf.constant([[0, 0], [1, 2]], dtype=tf.int64)
    values = tf.constant([1.0, 2.0], dtype=tf.float32)
    dense_shape = tf.constant([2, 3], dtype=tf.int64)
    sparse_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    
    # Test with training=True (uses smart_cond)
    output = dropout(sparse_input, training=True)
    assert isinstance(output, tf.SparseTensor)
    
    # Test with training=False (uses smart_cond)
    output = dropout(sparse_input, training=False)
    assert isinstance(output, tf.SparseTensor)


def test_transformer_encoder():
    """Test that TransformerEncoder works with Keras 3.0."""
    encoder = TransformerEncoder(
        num_layers=2,
        units=64,
        num_heads=4,
        dropout_rate=0.1,
    )
    
    # Create dummy input
    inputs = tf.random.normal((2, 10, 64))
    mask = tf.ones((2, 10), dtype=tf.bool)
    
    output = encoder(inputs, mask=mask, training=True)
    assert output.shape == (2, 10, 64)


def test_rasa_model_with_tf_utils():
    """Test that RasaModel works with the new tf_utils implementation."""
    # Create a minimal model data
    model_data = RasaModelData(
        label_key=LABEL,
        label_sub_key=IDS,
        data={
            TEXT: {
                SENTENCE: [
                    np.random.rand(5, 10).astype(np.float32)
                ]
            }
        },
    )
    
    # Create a simple model subclass
    class SimpleRasaModel(RasaModel):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(1)
        
        def batch_loss(self, batch_in):
            return tf.constant(0.0)
        
        def batch_predict(self, batch_in):
            x = batch_in[0]
            return {'output': self.dense(x)}
    
    model = SimpleRasaModel()
    
    # Test that the model can be compiled
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        run_eagerly=True
    )
    
    # Test prediction (uses tf_utils.sync_to_numpy_or_python_type)
    test_data = RasaModelData(
        label_key=LABEL,
        label_sub_key=IDS,
        data={
            TEXT: {
                SENTENCE: [
                    np.random.rand(2, 10).astype(np.float32)
                ]
            }
        },
    )
    
    # This should work without errors
    output = model.run_inference(test_data, batch_size=1)
    assert 'output' in output
    assert isinstance(output['output'], np.ndarray)


def test_model_save_load_compatibility():
    """Test that model saving/loading works with TensorFlow 2.16+."""
    import tempfile
    import os
    
    class SimpleRasaModel(RasaModel):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(1)
        
        def batch_loss(self, batch_in):
            return tf.constant(0.0)
        
        def batch_predict(self, batch_in):
            x = batch_in[0]
            return {'output': self.dense(x)}
    
    model = SimpleRasaModel()
    
    # Build the model
    dummy_input = tf.random.normal((1, 10))
    _ = model.batch_predict((dummy_input,))
    
    # Test saving weights (uses save_format="tf")
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "test_weights")
        model.save_weights(weights_path, save_format="tf")
        
        # Verify file was created
        assert os.path.exists(weights_path + ".index")
        
        # Test loading weights
        new_model = SimpleRasaModel()
        dummy_input2 = tf.random.normal((1, 10))
        _ = new_model.batch_predict((dummy_input2,))
        new_model.load_weights(weights_path)


def test_keras_backend_compatibility():
    """Test that Keras backend operations work correctly."""
    from tensorflow.keras import backend as K
    
    # Test basic operations
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    result = K.sum(x)
    assert tf.is_tensor(result)
    assert result.numpy() == 10.0


def test_tensorflow_version_compatibility():
    """Test that we're using a compatible TensorFlow version."""
    tf_version = tf.__version__
    version_parts = tf_version.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    # Should be TensorFlow 2.16.0 or higher
    assert major == 2, f"Expected TensorFlow 2.x, got {tf_version}"
    assert minor >= 16, f"Expected TensorFlow >= 2.16.0, got {tf_version}"


def test_keras_model_inheritance():
    """Test that RasaModel properly inherits from Keras Model."""
    from rasa.utils.tensorflow.models import RasaModel
    
    # Verify inheritance
    assert issubclass(RasaModel, tf.keras.Model)
    
    # Test instantiation
    model = RasaModel()
    assert isinstance(model, tf.keras.Model)


def test_smart_cond_in_layers():
    """Test that smart_cond is used correctly in layer implementations."""
    from rasa.utils.tensorflow.compat import smart_cond
    
    # Test that smart_cond works as expected
    condition = tf.constant(True)
    
    def true_fn():
        return tf.constant(1.0)
    
    def false_fn():
        return tf.constant(0.0)
    
    result = smart_cond(condition, true_fn, false_fn)
    assert result.numpy() == 1.0
    
    # Test with False condition
    condition_false = tf.constant(False)
    result = smart_cond(condition_false, true_fn, false_fn)
    assert result.numpy() == 0.0

