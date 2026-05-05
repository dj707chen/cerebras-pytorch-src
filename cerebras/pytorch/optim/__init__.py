# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Contains all Cerebras compliant Optimizer classes."""
import inspect
import logging
from copy import deepcopy

import numpy
import torch

from cerebras.appliance.utils.classes import retrieve_all_subclasses

from . import lr_scheduler, scheduler, weight_decay_scheduler
from .Adadelta import Adadelta
from .Adafactor import Adafactor
from .Adagrad import Adagrad
from .Adamax import Adamax
from .AdamBase import Adam, AdamW
from .ASGD import ASGD
from .Lamb import Lamb
from .Lion import Lion
from .NAdam import NAdam
from .optimizer import Optimizer
from .RAdam import RAdam
from .RMSprop import RMSprop
from .Rprop import Rprop
from .SGD import SGD


def configure_optimizer_params(optimizer_type: str, kwargs: dict):
    """
    Configures and requires an Optimizer specified using the provided optimizer
    type

    The optimizer class's signature is inspected and relevant parameters
    are extracted from the keyword arguments.

    Args:
        optimizer_type: The name of the optimizer to configure
        kwargs: Flattened optimizer params
    Returns:
        Optimizer cls, and args for initialization
    """

    optimizer_map = {
        cls.__name__.lower(): cls for cls in retrieve_all_subclasses(Optimizer)
    }

    if optimizer_type.lower() not in optimizer_map:
        raise ValueError(
            f"Invalid optimizer type. Expected one of "
            f"{sorted(optimizer_map.keys())}. Got: {optimizer_type}"
        )

    cls = optimizer_map[optimizer_type.lower()]

    learning_rate = kwargs.pop("learning_rate", None)
    if isinstance(learning_rate, (float, str)):
        learning_rate = float(learning_rate)
    else:
        learning_rate = None

    # common aliases
    aliases = {
        "weight_decay": ["weight_decay_rate"],
        "betas": ["beta1", "beta2"],
        "eps": ["eps1", "eps2"],
        "etas": ["eta1", "eta2"],
        "step_sizes": ["step_size_min", "step_size_max"],
    }

    deprecated_params = []
    cls_kwargs = {}
    # inspect the optimizer and extract the required parameters from the kwargs
    signature = inspect.signature(cls.__init__)
    for idx, (name, parameter) in enumerate(signature.parameters.items()):
        if idx == 0:
            pass  # "self"
        elif idx == 1:
            if name != "params":
                raise ValueError(
                    f"To use \"cstorch.optim.configure_optimizer()\" API, the optimizer class "
                    f"must accept \"params\" as the first argument after self, but the constructor "
                    f"signature is: {cls.__name__}{signature}"
                )
        # pylint: disable=protected-access
        elif parameter.kind == inspect._ParameterKind.VAR_KEYWORD:
            cls_kwargs.update(kwargs)
            break
        elif name in kwargs:
            cls_kwargs[name] = kwargs.pop(name)
        elif name in ("lr", "learning_rate"):
            if learning_rate is None:
                if parameter.default is not inspect.Parameter.empty:
                    learning_rate = parameter.default
                else:
                    learning_rate = 0.1  # default dummy value
            cls_kwargs[name] = learning_rate
        elif name in aliases:
            for alias in aliases[name]:
                if isinstance(alias, str) and alias in kwargs:
                    deprecated_params.append((name, alias))
                    break
                elif isinstance(alias, (list, tuple)) and all(
                    a in kwargs for a in alias
                ):
                    deprecated_params.append(
                        (name, str(alias).replace("'", ""))
                    )
                    break

    if deprecated_params:
        error_str = (
            f"{cls.__name__} got the following parameters which are no longer supported. "
            f"Please update your configs to use the new parameters instead."
        )
        for expected, actual in deprecated_params:
            error_str += f"\n\texpected: {expected}, actual: {actual}"
        raise ValueError(error_str)

    if len(kwargs) > 0:
        # Replace the default values in the signature to show the user the
        # values they passed in in the warning message so that they can verify
        # what they actually passed in
        signature = signature.replace(
            parameters=[
                inspect.Parameter(
                    name=name,
                    kind=param.kind,
                    default=cls_kwargs.get(
                        name,
                        (
                            param.default
                            if (
                                param.default != inspect.Parameter.empty
                                or name == "params"
                            )
                            else "<missing>"
                        ),
                    ),
                    annotation=param.annotation,
                )
                for name, param in list(signature.parameters.items())[1:]
            ]
        )
        logging.warning(
            f"{cls.__name__} got {len(kwargs)} unexpected "
            f"and unused parameters: {sorted(kwargs.keys())}.\n"
            f"Please ensure that you specified the correct parameters:\n"
            f"{cls.__name__}{signature}\n"
            f"Passing in unused parameters is deprecated behaviour and "
            f"support for it will be removed in a future release."
        )
    return cls, cls_kwargs


def configure_optimizer(optimizer_type: str, params, **kwargs):
    """
    Configures and requires an Optimizer specified using the provided optimizer
    type

    The optimizer class's signature is inspected and relevant parameters
    are extracted from the keyword arguments

    Args:
        optimizer_type: The name of the optimizer to configure
        params: The model parameters passed to the optimizer
    """
    cls, cls_kwargs = configure_optimizer_params(
        optimizer_type=optimizer_type, kwargs=kwargs
    )
    try:
        return cls(params, **cls_kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"Failed to configure {cls.__name__} optimizer"
        ) from e


def configure_scheduler_params(learning_rate: dict):
    """
    Get the kwargs and LR class from params

    Args:
        learning_rate (dict): learning rate config

    Returns:
        cls, kw_args : LR class and args
    """

    lr_scheduler_map = {
        cls.__name__.lower(): cls
        for cls in retrieve_all_subclasses(lr_scheduler.LRScheduler)
    }
    scheduler = learning_rate.pop("scheduler").lower()
    for name in (scheduler, f"{scheduler}lr"):
        if name in lr_scheduler_map:
            cls = lr_scheduler_map[name]
            break
    else:
        raise ValueError(
            f"Invalid lr_scheduler type. Expected one of "
            f"{list(lr_scheduler_map.keys())}. Got: {scheduler}"
        )
    # common aliases
    aliases = {
        "total_iters": ["steps", "decay_steps"],
        "initial_learning_rate": ["learning_rate", "base_lr"],
        "base_lr": ["learning_rate", "initial_learning_rate"],
        "learning_rates": ["values"],
        "milestones": ["boundaries"],
    }
    deprecated_params = []
    cls_kwargs = {}
    has_arbitrary_kwargs = False
    # inspect the optimizer and extract the required parameters from the kwargs
    signature = inspect.signature(cls.__init__)
    for idx, (name, parameter) in enumerate(signature.parameters.items()):
        if idx == 0:
            pass  # self
        elif idx == 1:
            if name != "optimizer":
                raise ValueError(
                    f"To use \"cstorch.optim.configure_lr_scheduler()\" API, the LR class "
                    f"must accept \"optimizer\" as the first argument after self, but the "
                    f"constructor signature is: {cls.__name__}{signature}"
                )
        # pylint: disable=protected-access
        elif parameter.kind == inspect._ParameterKind.VAR_KEYWORD:
            cls_kwargs.update(learning_rate)
            has_arbitrary_kwargs = True
            break
        elif name in learning_rate:
            cls_kwargs[name] = learning_rate.pop(name)
        elif name.lower() in learning_rate:
            deprecated_params.append((name, name.lower()))
        elif name in aliases:
            for alias in aliases[name]:
                if alias in learning_rate:
                    deprecated_params.append((name, alias))
                    break
    if deprecated_params:
        error_str = (
            f"{cls.__name__} got the following parameters which are no longer supported. "
            f"Please update your configs to use the new parameters."
        )
        for expected, actual in deprecated_params:
            error_str += f"\n\texpected: {expected}, actual: {actual}"
        raise ValueError(error_str)

    if len(learning_rate) > 0 and not has_arbitrary_kwargs:
        # Replace the default values in the signature to show the user the
        # values they passed in in the warning message so that they can verify
        # what they actually passed in
        signature = signature.replace(
            parameters=[
                inspect.Parameter(
                    name=name,
                    kind=param.kind,
                    default=cls_kwargs.get(
                        name,
                        (
                            param.default
                            if (
                                param.default != inspect.Parameter.empty
                                or name == "optimizer"
                            )
                            else "<missing>"
                        ),
                    ),
                    annotation=param.annotation,
                )
                for name, param in list(signature.parameters.items())[1:]
            ]
        )
        raise ValueError(
            f"{cls.__name__} got {len(learning_rate)} unexpected "
            f"parameters: {sorted(learning_rate.keys())}.\n"
            f"Please ensure that you specified the correct parameters:\n"
            f"{cls.__name__}{signature}\n"
        )
    return cls, cls_kwargs


def get_scheduler(optimizer: Optimizer, learning_rate: dict):
    """
    Gets the LR scheduler from learning rate dict

    Args:
        learning_rate (dict): learning rate config

    Returns:
        cls : LRScheduler class obj
    """

    cls, cls_kwargs = configure_scheduler_params(learning_rate)
    try:
        return cls(optimizer, **cls_kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"Failed to configure {cls.__name__} scheduler"
        ) from e


def configure_lr_scheduler(optimizer, learning_rate, adjust_learning_rate=None):
    """
    Configures a learning rate scheduler specified using the provided lr_scheduler
    type

    The learning rate scheduler's class's signature is inspected and relevant
    parameters are extracted from the keyword arguments

    Args:
        optimizer: The optimizer passed to the lr_scheduler
        learning_rate: learning rate schedule
        adjust_learning_rate (dict): key: layer types, val: lr scaling factor
    """

    if not learning_rate:
        return None

    learning_rate = deepcopy(learning_rate)

    if isinstance(learning_rate, (float, str)):
        learning_rate_dicts = [
            {"scheduler": "constant", "learning_rate": float(learning_rate)}
        ]

    elif isinstance(learning_rate, dict):
        learning_rate_dicts = [learning_rate]

    elif isinstance(learning_rate, (list, tuple)):
        learning_rate_dicts = learning_rate

    else:
        raise ValueError(
            f"Unsupported LR scheduler type."
            f"Expected one of float/dict/list/tuple. "
            f"Got: {learning_rate}"
        )

    schedulers = []
    main_scheduler = set()
    for params in learning_rate_dicts:
        # TODO: figure out a better way to specify this
        if sched := params.pop("main_scheduler", None):
            main_scheduler.add(sched)
        schedulers.append(get_scheduler(optimizer, params))
    if len(main_scheduler) > 1:
        raise ValueError(
            f"Got conflicting `main_scheduler` values: {main_scheduler}. "
            f"Please make sure to specify the same main scheduler."
        )
    main_scheduler = list(main_scheduler)[0] if main_scheduler else None

    if main_scheduler is not None and main_scheduler.lower() in (
        "chained",
        "chainedlr",
    ):
        if adjust_learning_rate:
            raise ValueError(
                f"{main_scheduler} is not supported with lr scaling. Either use non-chained "
                "scheduler or don't use lr scaling (don't pass in `adjust_learning_rate`)"
            )
        return lr_scheduler.ChainedScheduler(schedulers)

    if len(schedulers) == 1:
        if adjust_learning_rate:
            return lr_scheduler.ScalePerParamLR(optimizer, schedulers[0])
        else:
            return schedulers[0]

    # default to sequential
    total_iters = [scheduler.total_iters for scheduler in schedulers[:-1]]
    assert all(total_iter is not None for total_iter in total_iters)

    milestones = numpy.array(total_iters).cumsum().tolist()

    if adjust_learning_rate:
        return lr_scheduler.ScalePerParamLR(
            optimizer,
            lr_scheduler.SequentialLR(optimizer, schedulers, milestones),
        )
    else:
        return lr_scheduler.SequentialLR(optimizer, schedulers, milestones)


def _get_milestones(schedulers):
    total_iters = [scheduler.total_iters for scheduler in schedulers[:-1]]
    assert all(total_iter is not None for total_iter in total_iters)

    milestones = numpy.array(total_iters).cumsum().tolist()
    return milestones


def configure_scheduler(optimizer, schedulers_params: dict):
    """
    Configures a generic scheduler from scheduler params.
    The scheduler class' signature is inspected and relevant
    parameters are extracted from the keyword arguments.

    Args:
        optimizer: The optimizer passed to each scheduler.
        schedulers_params: A dict of scheduler params.
    """

    scheduler_params = deepcopy(schedulers_params)

    scheduler_name = next(iter(schedulers_params))

    scheduler_cls = None

    scheduler_map = {
        cls.__name__: cls
        for cls in retrieve_all_subclasses(scheduler.Scheduler)
    }
    scheduler_cls = scheduler_map.get(scheduler_name, None)

    if scheduler_cls is None:
        raise ValueError(
            f"Invalid scheduler type. Expected one of "
            f"{sorted(scheduler_map)}. Got: ({scheduler_name})"
        )

    def _configure_schedulers(
        optimizer,
        scheduler_params_list,
        main_scheduler_name,
        param_group_tags=None,
    ):
        # param_group_tags is passed as an argument to overwrite param_group_tags
        # in the sub-schedulers
        _schedulers = []
        for _scheduler_params in scheduler_params_list:
            name = next(iter(_scheduler_params))
            if "param_group_tags" in _scheduler_params[name]:
                raise ValueError(
                    f"Parameter `param_group_tags` found in a non-main scheduler "
                    f"{name}. A {main_scheduler_name} main-scheduler expects all "
                    f"sub-schedulers to have no `param_group_tags`."
                )
            if param_group_tags:
                _scheduler_params[name]["param_group_tags"] = param_group_tags
            _schedulers.append(
                configure_scheduler(optimizer, _scheduler_params)
            )
        return _schedulers

    if issubclass(
        scheduler_cls,
        scheduler.SequentialScheduler,
    ) and not issubclass(scheduler_cls, scheduler.PiecewiseConstantScheduler):
        # recursively call configure_scheduler to get schedulers list
        param_group_tags = schedulers_params[scheduler_name].get(
            "param_group_tags", None
        )
        schedulers = _configure_schedulers(
            optimizer,
            schedulers_params[scheduler_name].pop("schedulers"),
            scheduler_name,
            param_group_tags,
        )
        milestones = _get_milestones(schedulers)
        return scheduler_cls(
            optimizer,
            schedulers,
            milestones,
            **schedulers_params[scheduler_name],
        )
    elif issubclass(
        scheduler_cls,
        scheduler.ChainedScheduler,
    ):
        # recursively call configure_scheduler to get schedulers list
        param_group_tags = schedulers_params[scheduler_name].get(
            "param_group_tags", None
        )
        schedulers = _configure_schedulers(
            optimizer,
            schedulers_params[scheduler_name].pop("schedulers"),
            scheduler_name,
            param_group_tags,
        )
        return scheduler_cls(schedulers, **schedulers_params[scheduler_name])
    elif issubclass(
        scheduler_cls,
        scheduler.ScalePerParamScheduler,
    ):
        # recursively call configure_scheduler to get schedulers list
        param_group_tags = schedulers_params[scheduler_name].get(
            "param_group_tags", None
        )
        sub_scheduler = _configure_schedulers(
            optimizer,
            [schedulers_params[scheduler_name].pop("scheduler")],
            scheduler_name,
            param_group_tags,
        )
        if len(sub_scheduler) != 1:
            raise ValueError(
                f"{scheduler_name} must contain exactly 1 scheduler."
            )
        return scheduler_cls(
            optimizer, sub_scheduler[0], **schedulers_params[scheduler_name]
        )
    else:
        return scheduler_cls(optimizer, **schedulers_params[scheduler_name])


__all__ = [
    "Optimizer",
    # Subclasses must be listed here for docs to be autogenerated
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adamax",
    "Adam",
    "AdamW",
    "ASGD",
    "Lamb",
    "Lion",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "configure_optimizer",
    "configure_optimizer_params",
    "configure_scheduler_params",
    "configure_lr_scheduler",
    "configure_scheduler",
    "scheduler",
    "lr_scheduler",
    "weight_decay_scheduler",
]
