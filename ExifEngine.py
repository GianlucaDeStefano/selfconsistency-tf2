import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
from Detectors.DetectorEngine import DetectorEngine
from Detectors.Exif import demo


class ExifEngine(DetectorEngine):
    weights_path = os.path.join(pathlib.Path(__file__).parent, './ckpt/exif_final/exif_final.ckpt')

    def __init__(self, dense=False):
        tf.compat.v1.disable_eager_execution()
        super().__init__("ExifEngine")
        self.model = demo.Demo(ckpt_path=self.weights_path, use_gpu=0, quality=3.0, num_per_dim=30)
        self.dense = dense

    def destroy(self):
        """
        @return:
        """
        # check if a session object exists
        if self.model.solver.sess:

            # close tf section if open
            if not self.model.solver.sess._closed:
                self.model.solver.sess.close()

        del self.model

    def reset(self):
        """
        Reset Exif to a pristine state
        @return: None
        """
        self.metadata = dict()

        if not self.model.solver.sess or self.model.solver.sess._closed:
            self.model = demo.Demo(ckpt_path=self.weights_path, use_gpu=0, quality=3.0, num_per_dim=30)

    def initialize(self, sample_path, reset_instance=True):
        """
        Initialize the detector to handle a new sample.

        @param sample_path: str
            Path of the sample to analyze
        @param reset_instance: Bool
            A flag indicating if this detector's metadata should be reinitialized before loading the new sample
        """

        # Generic initialization of the detector engine
        super(ExifEngine, self).initialize(sample_path, reset_instance)

        # Make sure the necessary data has been loaded
        assert ("sample_path" in self.metadata.keys())

        # read the necessary metadata
        sample_path = self.metadata["sample_path"]

        # Load the sample
        sample = cv2.imread(sample_path)[:, :, [2, 1, 0]]
        self.metadata["sample"] = sample

    def extract_features(self):
        # check if the featuresvhave already been extracted
        if "features" in self.metadata:
            return self.metadata

        # Make sure the necessary data has been loaded
        assert ("sample" in self.metadata.keys())

        # read the necessary metadata
        sample = self.metadata["sample"]

        if self.dense:
            self.metadata["features"] = self.model.run_vote_extract_features(sample)
        else:
            self.metadata["features"] = self.model.run_extract_features(sample)

        return self.metadata

    def process_features(self, compute_mask=False):
        # check if the features have already been processed
        if "heatmap" in self.metadata:
            return self.metadata

        # read the necessary metadata
        sample = self.metadata["sample"]
        features = self.metadata["features"]

        # compute the heatmap
        if self.dense:
            self.metadata["heatmap"] = self.model.run_vote_cluster(features)[0]
        else:
            self.metadata["heatmap"] = self.model.run_cluster_heatmap(sample, features, False)

            if compute_mask:
                self.metadata["mask"] = self.model.run_cluster_mask(sample, features)

        if np.mean(self.metadata["heatmap"] > 0.5) > 0.5:
            self.metadata["heatmap"] = 1.0 - self.metadata["heatmap"]

        return self.metadata
