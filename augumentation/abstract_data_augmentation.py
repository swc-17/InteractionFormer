from abc import ABC, abstractmethod
from typing import List


class AbstractAugmentor(ABC):
    """Augmentor interface used in applying data augmentation techniques to input features/targets during training."""

    def validate(self, features, targets) -> None:
        """
        Checks whether the required feature/target tensor keys match the input feature/target tensor keys.
        :param features: Input feature tensors to be validated.
        :param targets: Input target tensors to be validated.
        """
        required_features = set(self.required_features)
        present_features = set(features.keys())
        assert len(required_features - present_features) == 0, (
            f'Augmentor requires features {required_features} but invoked with '
            f'features dict that contains features {present_features}'
        )

        required_target = set(self.required_targets)
        present_target = set(targets.keys())
        assert len(required_target - present_target) == 0, (
            f'Augmentor requires target {required_target} but invoked with '
            f'targets dict that contains target {present_target}'
        )

    @abstractmethod
    def augment(
        self, features, targets):
        """
        Run augmentation against the input feature and target tensors.
        :param features: Input feature tensors to be augmented.
        :param targets: Input target tensors to be augmented.
        :param scenario: The scenario where features and targets are generated from.
        :return: Augmented features and targets.
        """
        pass

    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        """List of required features by the augmentor."""
        pass

    @property
    @abstractmethod
    def required_targets(self) -> List[str]:
        """List of required targets by the augmentor."""
        pass

    @property
    @abstractmethod
    def augmentation_probability(self):
        """
        Augmentation probability
        :return: Augmentation probability of the augmentor.
        """
        pass

    @property
    def get_schedulable_attributes(self):
        """
        Gets attributes to be modified by augmentation scheduler callback.
        :return: Attributes to be modified by augmentation scheduler callback.
        """
        return []
