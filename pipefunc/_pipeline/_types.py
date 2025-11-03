from typing import Literal, TypeAlias

OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]
DefaultStorageTypes: TypeAlias = Literal["file_array", "dict", "shared_memory_dict"]
StorageType: TypeAlias = str | DefaultStorageTypes | dict[OUTPUT_TYPE, DefaultStorageTypes | str]
