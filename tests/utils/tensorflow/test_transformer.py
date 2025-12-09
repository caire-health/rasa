import pytest
import tensorflow as tf

from rasa.utils.tensorflow.transformer import MultiHeadAttention


def test_valid_transformer_size():
    mha = MultiHeadAttention(units=256, num_heads=4)
    assert mha.units == 256
    with pytest.raises(SystemExit):
        MultiHeadAttention(units=50, num_heads=4)


def test_multi_head_attention_with_training():
    """Test MultiHeadAttention with training flag (uses smart_cond)."""
    mha = MultiHeadAttention(units=64, num_heads=4)
    
    query = tf.random.normal((2, 10, 64))
    key = tf.random.normal((2, 10, 64))
    value = tf.random.normal((2, 10, 64))
    
    # Test with training=True
    output_train = mha(query, key, value, training=True)
    assert output_train.shape == (2, 10, 64)
    
    # Test with training=False
    output_eval = mha(query, key, value, training=False)
    assert output_eval.shape == (2, 10, 64)
    
    # Both should produce valid outputs
    assert tf.is_tensor(output_train)
    assert tf.is_tensor(output_eval)
