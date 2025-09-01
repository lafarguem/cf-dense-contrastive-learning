class DualBranchContrastiveOutput:
    def __init__(
            self, 
            dense_embeddings = None, 
            teacher_dense_embeddings = None, 
            instance_embeddings = None, 
            teacher_instance_embeddings=None,
            projected_dense_embeddings = None,
            projected_instance_embeddings = None,
            projected_teacher_dense_embeddings = None,
            projected_teacher_instance_embeddings = None,
            coords = None,
        ):
        self.dense_embeddings = dense_embeddings
        self.teacher_dense_embeddings = teacher_dense_embeddings
        self.instance_embeddings = instance_embeddings
        self.teacher_instance_embeddings = teacher_instance_embeddings
        self.projected_dense_embeddings = projected_dense_embeddings
        self.projected_instance_embeddings = projected_instance_embeddings
        self.projected_teacher_dense_embeddings = projected_teacher_dense_embeddings
        self.projected_teacher_instance_embeddings = projected_teacher_instance_embeddings
        self.coords = coords
        if dense_embeddings is not None:
            self.num_views = dense_embeddings.shape[1]
        else:
            self.num_views = instance_embeddings.shape[1]

class SingleBranchContrastiveOutput:
    def __init__(
            self, 
            dense_embeddings = None, 
            instance_embeddings = None,
            projected_dense_embeddings = None,
            projected_instance_embeddings = None,
            coords = None,
        ):
        self.dense_embeddings = dense_embeddings
        self.instance_embeddings = instance_embeddings
        self.projected_dense_embeddings = projected_dense_embeddings
        self.projected_instance_embeddings = projected_instance_embeddings
        self.coords = coords
        if dense_embeddings is not None:
            self.num_views = dense_embeddings.shape[1]
        else:
            self.num_views = instance_embeddings.shape[1]