import torch

def rodrigues_batch(rvecs):
    """
    Pytorch implementation of: https://github.com/blzq/tf_rodrigues/blob/master/rodrigues.py

    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, frames, num_rots, 3) to a batch of rotation matrices shaped (batch, frames, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    # batch_size = tf.shape(rvecs)[0]
    # frames = tf.shape(rvecs)[1]
    # num_rots = tf.shape(rvecs)[2]
    # vec_size = tf.shape(rvecs)[3]
    # tf.assert_equal(vec_size, 3)
    batch_size, frames, num_rots, vec_size = rvecs.shape
    assert vec_size == 3

    # thetas = tf.norm(rvecs, axis=3, keepdims=True)
    thetas = torch.norm(rvecs, dim=3, keepdim=True)
    # is_zero = tf.equal(tf.squeeze(thetas, axis=-1), 0.0)
    # is_zero = tf.expand_dims(is_zero, axis=3)
    # is_zero = tf.expand_dims(is_zero, axis=3)
    is_zero = thetas.squeeze(-1) == 0
    is_zero = is_zero.unsqueeze(3).unsqueeze(3)
    u = rvecs / thetas
    
    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.zeros((batch_size, frames, num_rots), device=rvecs.device)  # for broadcasting
    Ks_1 = torch.stack((  zero   , -u[:, :, :, 2],  u[:, :, :, 1]), dim=3)  # row 1
    Ks_2 = torch.stack((  u[:, :, :, 2],  zero   , -u[:, :, :, 0]), dim=3)  # row 2
    Ks_3 = torch.stack(( -u[:, :, :, 1],  u[:, :, :, 0],  zero   ), dim=3)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], dim=3)                  # stack rows

    # Rs = tf.eye(3, batch_shape=[batch_size, frames, num_rots]) + \
    #      tf.sin(thetas)[..., tf.newaxis] * Ks + \
    #      (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)
    multi_dim_eye_mtx = torch.eye(3, device=rvecs.device).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat((batch_size, frames, num_rots, 1, 1))
    Rs = multi_dim_eye_mtx + \
        torch.sin(thetas).unsqueeze(-1) * Ks + \
        (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    # return tf.where(is_zero,
    #                 tf.eye(3, batch_shape=[batch_size, frames, num_rots]), Rs)
    return torch.where(is_zero, multi_dim_eye_mtx, Rs)
