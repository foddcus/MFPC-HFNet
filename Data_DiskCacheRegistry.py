# English counterpart comments added on 2026-06-18 by GG; original Chinese annotations are retained.
import json
import os
import shutil
import time
import hashlib
from typing import Dict, Optional


# ================= 磁盘缓存数据库管理器 =================
# EN: ================= disk cachedata database =================.
# 逻辑：
# EN: Logic:
# 1. 该模块专门负责维护预处理数据库的“索引管理文件”；
# EN: this maintain data database "";
# 2. 每一套训练数据库（由原始样本根目录、真值文件、目标模式、分辨率、NIR 维度、图像通道配置共同决定）
# EN: each traindata database (by start sampleroot directory, ground-truth file, form,, NIR degree, image pass same)
#    都会得到一个稳定的 dataset_signature；
# EN: all will obtain stable dataset_signature;
# 3. 索引文件会记录该 signature 对应的磁盘缓存目录、目标模式、期望样本数、最近一次校验结果等信息；
# EN: indexingfile will this signature for should disk cache, form, sample count, most result result;
# 4. 新训练启动时，优先通过索引文件寻找历史缓存目录，并先按“就绪样本数量”做快速核对；
# EN: new trainstart when, prefer pass indexingfile find cache, and first by "" do for;
# 5. 若索引失效、目录丢失、文件损坏或样本数量不一致，则判定该旧数据库无效：
# EN: if indexing,, file or sample count amount not consistent, then this old data database no:
#    自动删除旧目录，并为后续重建腾出干净位置；
# EN: automatically old, and as later;
# 6. 当新的磁盘缓存建成后，再把最新状态回写到索引文件，这样后续训练就无需手动填写缓存路径。
# EN: new disk cache after, then most new write to indexingfile, latertrain then no need manual write cachepath.
# 7. 最近修改时间：2026-05-20；默认缓存根目录由调用方迁移到 磁盘数据库目录，本模块仅负责通用索引管理。
# EN: Last modified: 2026-05-20; defaultcacheroot directory by use migrate to data database, only generalindexing.
# 8. 最近注释维护：2026-05-29；作者：ljy。补充函数级输入输出和失效缓存处理说明，不改变缓存判定逻辑。
# EN: Latest comment maintenance: 2026-05-29; Author: ljy. function inputs and outputs and cache note, does not changecache logic.


def _safe_norm(value) -> str:
    """
    将路径或文本字段规范化为稳定签名用字符串。
    English: pathfieldnormalize.

    输入:
    English: Input:
        value: 任意可转为字符串的路径/字段；None 会被视为空串。
        English: value: path/field; None .
    输出:
    English: Output:
        小写、统一斜杠方向后的字符串。
        English: , .

    设计说明:
    English: Design note:
        dataset_signature 需要跨 Windows 路径写法保持稳定，因此这里统一大小写和分隔符。
        English: dataset_signature Windows path, .
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    if value is None:
        return ''
    return os.path.normpath(str(value)).replace('\\', '/').lower()


def build_dataset_signature(
    data_root,
    gt_path,
    tn_path,
    image_size,
    nir_dim,
    target_mode,
    image_channels=None,
    wavelengths=None,
    active_inputs=None,
) -> str:
    """
    构建训练数据库级别的稳定签名。
    English: buildtraining.

    输入:
    English: Input:
        data_root: 原始样本库根目录。
        English: data_root: sampledirectory.
        gt_path / tn_path: SOC/TN 真值文件路径。
        English: tn_path: SOC/TN 真值文件路径.
        image_size: 预处理后的图像尺寸；None 表示原始尺寸。
        English: image_size: image; None .
        nir_dim: NIR 特征维度。
        English: nir_dim: NIR .
        target_mode: 训练目标模式，通常为 soc/tn/both。
        English: target_mode: training target mode, soc/tn/both.
        image_channels / wavelengths: 图像通道数和波长列表，用于区分 8 通道数据库配置。
        English: wavelengths: 图像通道数和波长列表，用于区分 8 通道数据库配置.
        active_inputs: 当前训练实际启用的输入源，用于区分只读 NIR/Hyper 缓存与全模态缓存。
        English: active_inputs: currenttrainingInput, NIR/Hyper cachecache.
    输出:
    English: Output:
        SHA1 十六进制字符串，用作 registry 中的一套数据缓存身份。
        English: SHA1 , registry cache.

    设计说明:
    English: Design note:
        签名覆盖“数据来源 + 标签来源 + 输入形态”，避免不同任务误复用同一磁盘缓存。
        English: “ + label + Input”, avoidcache.
        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
    """
    image_tag = 'orig' if image_size is None else f"{int(image_size[0])}x{int(image_size[1])}"
    wavelengths_tag = ''
    if wavelengths is not None:
        wavelengths_tag = ','.join([str(w).strip() for w in wavelengths])
    active_inputs_tag = ''
    if active_inputs is not None:
        active_inputs_tag = '+'.join(sorted(str(item).strip().lower() for item in active_inputs if str(item).strip()))

    raw = '||'.join([
        _safe_norm(data_root),
        _safe_norm(gt_path),
        _safe_norm(tn_path),
        str(image_tag),
        str(int(nir_dim)),
        str(target_mode).strip().lower(),
        '' if image_channels is None else str(int(image_channels)),
        wavelengths_tag,
        active_inputs_tag,
    ])
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


class DiskCacheRegistryManager:
    """
    磁盘预处理磁盘缓存的索引管理器。
    English: cache.

    职责:
    English: :
        1. 读取/保存 disk_cache_registry.json；
        English: 1. read/save disk_cache_registry.json;
        2. 根据 dataset_signature 查找历史缓存目录；
        English: 2. dataset_signature cachedirectory;
        3. 轻量检查缓存目录中就绪样本数量；
        English: 3. checkcachedirectorysample;
        4. 当旧缓存损坏或样本数不匹配时清理旧目录，防止训练误读半成品。
        English: 4. cachesampledirectory, training.

    最近修改时间：2026-05-29；作者：ljy。
    English: Last modified: 2026-05-29; Author: ljy.
    """

    def __init__(self, cache_root: str, registry_filename: str = 'disk_cache_registry.json'):
        """
        初始化管理器并加载索引文件。
        English: loadfile.

        输入:
        English: Input:
            cache_root: 磁盘缓存总目录。
            English: cache_root: cachedirectory.
            registry_filename: 管理文件名，默认 disk_cache_registry.json。
            English: registry_filename: file, default disk_cache_registry.json.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        self.cache_root = os.path.normpath(cache_root)
        os.makedirs(self.cache_root, exist_ok=True)
        self.registry_path = os.path.join(self.cache_root, registry_filename)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """
        读取并修复 registry 基本结构。
        English: read registry .

        输出:
        English: Output:
            至少包含 version、updated_at、datasets 三个字段的字典。
            English: version, updated_at, datasets fielddictionary.

        设计说明:
        English: Design note:
            管理文件损坏时不直接中断训练，而是返回空索引；旧缓存目录会在后续目录检查中重新判定。
            English: filetraining, return; cachedirectorydirectorycheck.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        if not os.path.isfile(self.registry_path):
            return {
                'version': 2,
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'datasets': {}
            }
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError('registry root is not dict')
            data.setdefault('version', 2)
            data.setdefault('updated_at', time.strftime('%Y-%m-%d %H:%M:%S'))
            data.setdefault('datasets', {})
            return data
        except Exception:
            # 若管理文件本身损坏，则直接重置；旧缓存目录仍会在后续扫描时被再次校验。
            # EN: if file, then directly; old cache still will in laterscan when then.
            return {
                'version': 2,
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'datasets': {}
            }

    def save(self):
        """
        将当前 registry 状态写回磁盘。
        English: current registry .

        说明:
        English: :
            每次保存前刷新 updated_at，便于判断管理文件最近一次由哪轮数据准备流程维护。
            English: save updated_at, determinefile.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        self.registry['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    # ================= 缓存目录数量核对函数 =================
    # EN: ================= cache count for function =================.
    # 逻辑：
    # EN: Logic:
    # 1. 这里不重新读取全部 numpy 数据，只做轻量级“目录 + 必要文件”检查；
    # EN: here not new read full numpy data, only do amount " + "check;
    # 2. 对每个样本缓存目录，要求至少同时存在当前 active_inputs 对应的 .npy 文件和 meta.json；
    # EN: for each samplecache, need fewer same when in current active_inputs for should.npy file and meta.json;
    # 3. 统计 ready_sample_count，并与 expected_sample_count 比较；
    # EN: ready_sample_count, and and expected_sample_count;
    # 4. 用户明确要求先核对 disk 数量，因此这里把 ready_sample_count 是否一致作为首要有效性判据。
    # EN: use confirm need first for disk count, thereforehere ready_sample_count is no consistent as need property data.
    def inspect_cache_dir(self, cache_data_dir: str, expected_sample_count: int, required_files=None) -> Dict:
        """
        轻量检查一个缓存目录是否完整。
        English: checkcachedirectory.

        输入:
        English: Input:
            cache_data_dir: 某套数据缓存的样本目录。
            English: cache_data_dir: cachesampledirectory.
            expected_sample_count: 本次扫描期望的有效样本数量。
            English: expected_sample_count: sample.
            required_files: 当前 active_inputs 对应的必要缓存文件名，默认检查全模态三件套。
            English: required_files: current active_inputs cachefile, defaultcheck.
        输出:
        English: Output:
            检查摘要字典，核心字段为 ready_sample_count 与 is_valid。
            English: checkdictionary, field ready_sample_count is_valid.

        注意:
        English: :
            这里不读取 .npy 内容，只检查每个样本目录的必要文件，避免在训练启动前产生额外内存压力。
            English: read .npy , checksampledirectoryfile, avoidtraining.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        cache_data_dir = os.path.normpath(cache_data_dir)
        required_files = tuple(required_files or ('hyper.npy', 'nir.npy', 'image.npy'))
        result = {
            'cache_data_dir': cache_data_dir,
            'exists': os.path.isdir(cache_data_dir),
            'expected_sample_count': int(expected_sample_count),
            'required_files': list(required_files),
            'ready_sample_count': 0,
            'is_valid': False,
            'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        if not result['exists']:
            return result

        ready_count = 0
        try:
            for entry in os.scandir(cache_data_dir):
                if not entry.is_dir():
                    continue
                sample_dir = entry.path
                if all(os.path.isfile(os.path.join(sample_dir, name)) for name in required_files) and os.path.isfile(
                    os.path.join(sample_dir, 'meta.json')
                ):
                    ready_count += 1
        except Exception:
            ready_count = -1

        result['ready_sample_count'] = int(max(ready_count, 0))
        result['is_valid'] = (ready_count >= 0 and int(ready_count) == int(expected_sample_count) and int(expected_sample_count) > 0)
        return result

    def delete_cache_dir(self, cache_data_dir: Optional[str]):
        """
        删除被判定无效的缓存目录。
        English: cachedirectory.

        输入:
        English: Input:
            cache_data_dir: 待删除的样本缓存目录；空值直接忽略。
            English: cache_data_dir: samplecachedirectory; empty value.

        设计说明:
        English: Design note:
            只在 resolve_cache_dir 确认旧缓存存在但不完整时调用，避免训练误复用半成品缓存。
            English: resolve_cache_dir cache, avoidtrainingcache.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        if not cache_data_dir:
            return
        cache_data_dir = os.path.normpath(cache_data_dir)
        if os.path.isdir(cache_data_dir):
            shutil.rmtree(cache_data_dir, ignore_errors=True)

    # ================= 通过管理文件解析磁盘数据库目录 =================
    # EN: ================= pass fileparse data database =================.
    # 逻辑：
    # EN: Logic:
    # 1. 优先读取 registry 中当前 dataset_signature 对应的历史目录；
    # EN: preferread registry in current dataset_signature for should;
    # 2. 若 registry 无记录，则回退到默认命名目录；
    # EN: if registry no, then fall back to default name;
    # 3. 对候选目录逐一执行数量核对，只要命中一个完整目录，就直接复用；
    # EN: for executecount for, only need in complete, then directlyreuse;
    # 4. 若候选目录存在但无效，则先删除旧目录，再把后续流程切到“重建”；
    # EN: if in no, then first old, then later program to "";
    # 5. 这样训练脚本无需手填缓存路径，是否存在历史数据库由管理文件自动判断。
    # EN: train no need cachepath, is no in data database by fileautomaticallydetermine.
    def resolve_cache_dir(
        self,
        dataset_signature: str,
        default_cache_data_dir: str,
        expected_sample_count: int,
        required_files=None,
    ) -> Dict:
        """
        解析当前训练任务应复用还是重建磁盘缓存。
        English: parsecurrenttrainingcache.

        输入:
        English: Input:
            dataset_signature: 由数据来源和输入配置生成的稳定签名。
            English: dataset_signature: Inputconfiguration.
            default_cache_data_dir: registry 未命中时的默认缓存目录。
            English: default_cache_data_dir: registry defaultcachedirectory.
            expected_sample_count: 本次训练扫描出的有效样本数量。
            English: expected_sample_count: trainingsample.
            required_files: 当前训练需要的 .npy 文件名；用于允许子模态任务复用全模态缓存。
            English: required_files: currenttraining .npy file; cache.
        输出:
        English: Output:
            决策摘要字典，包含 cache_data_dir、reuse_existing_cache、force_rebuild 等字段。
            English: dictionary, cache_data_dir, reuse_existing_cache, force_rebuild field.

        决策原则:
        English: :
            registry 命中且样本数完整时复用；否则删除无效旧目录并要求调用方重建。
            English: registry sample; directory.
            最近修改时间：2026-05-29；作者：ljy。
            English: Last modified: 2026-05-29; Author: ljy.
        """
        required_files = tuple(required_files or ('hyper.npy', 'nir.npy', 'image.npy'))
        datasets = self.registry.setdefault('datasets', {})
        entry = datasets.get(dataset_signature, {}) if isinstance(datasets.get(dataset_signature, {}), dict) else {}

        candidate_dirs = []
        if entry.get('cache_data_dir'):
            candidate_dirs.append(os.path.normpath(entry['cache_data_dir']))
        default_cache_data_dir = os.path.normpath(default_cache_data_dir)
        if default_cache_data_dir not in candidate_dirs:
            candidate_dirs.append(default_cache_data_dir)

        checked_results = []
        for candidate_dir in candidate_dirs:
            inspect_result = self.inspect_cache_dir(candidate_dir, expected_sample_count, required_files=required_files)
            inspect_result['source'] = 'registry' if candidate_dir == os.path.normpath(entry.get('cache_data_dir', '')) else 'default'
            checked_results.append(inspect_result)
            if inspect_result['is_valid']:
                return {
                    'cache_data_dir': candidate_dir,
                    'registry_entry_found': bool(entry),
                    'registry_hit': inspect_result['source'] == 'registry',
                    'reuse_existing_cache': True,
                    'force_rebuild': False,
                    'checked_results': checked_results,
                    'selected_check': inspect_result,
                }

        # 走到这里说明：历史目录不存在，或者存在但不完整。按用户要求，删除无效旧数据库并准备重建。
        # EN: to herenote: does not exist, or in not complete. by use need, no old data database and.
        for inspect_result in checked_results:
            if inspect_result['exists'] and not inspect_result['is_valid']:
                self.delete_cache_dir(inspect_result['cache_data_dir'])

        os.makedirs(default_cache_data_dir, exist_ok=True)
        return {
            'cache_data_dir': default_cache_data_dir,
            'registry_entry_found': bool(entry),
            'registry_hit': False,
            'reuse_existing_cache': False,
            'force_rebuild': True,
            'checked_results': checked_results,
            'selected_check': self.inspect_cache_dir(default_cache_data_dir, expected_sample_count, required_files=required_files),
        }

    def update_entry(
        self,
        dataset_signature: str,
        cache_data_dir: str,
        data_root,
        gt_path,
        tn_path,
        image_size,
        nir_dim,
        target_mode,
        expected_sample_count: int,
        ready_sample_count: int,
        resolved_cache_mode: str,
        image_channels: Optional[int] = None,
        wavelengths=None,
        active_inputs=None,
        required_files=None,
    ):
        """
        回写某套缓存数据库的最新状态。
        English: cache.

        输入:
        English: Input:
            dataset_signature: 当前训练数据库身份。
            English: dataset_signature: currenttraining.
            cache_data_dir: 实际使用的缓存目录。
            English: cache_data_dir: cachedirectory.
            data_root / gt_path / tn_path: 数据来源记录。
            English: gt_path / tn_path: 数据来源记录.
            image_size / nir_dim / target_mode: 输入和任务配置。
            English: nir_dim / target_mode: 输入和任务配置.
            expected_sample_count / ready_sample_count: 完整性统计。
            English: ready_sample_count: 完整性统计.
            resolved_cache_mode: 调用方最终使用的缓存模式。
            English: resolved_cache_mode: cache.
            image_channels / wavelengths: 图像通道配置记录。
            English: wavelengths: 图像通道配置记录.
            active_inputs / required_files: 当前训练实际启用输入源和缓存必要文件。
            English: required_files: 当前训练实际启用输入源和缓存必要文件.

        输出:
        English: Output:
            无返回值；函数会更新内存 registry 并写回 JSON。
            English: return; update registry JSON.

        最近修改时间：2026-05-29；作者：ljy。
        English: Last modified: 2026-05-29; Author: ljy.
        """
        image_tag = 'orig' if image_size is None else f"{int(image_size[0])}x{int(image_size[1])}"
        active_inputs_list = None if active_inputs is None else [str(item).strip().lower() for item in active_inputs if str(item).strip()]
        required_files_list = None if required_files is None else [str(item).strip() for item in required_files if str(item).strip()]
        datasets = self.registry.setdefault('datasets', {})
        old_entry = datasets.get(dataset_signature, {}) if isinstance(datasets.get(dataset_signature, {}), dict) else {}
        created_at = old_entry.get('created_at', time.strftime('%Y-%m-%d %H:%M:%S'))
        datasets[dataset_signature] = {
            'dataset_signature': dataset_signature,
            'cache_data_dir': os.path.normpath(cache_data_dir),
            'data_root': str(data_root),
            'gt_path': str(gt_path),
            'tn_path': '' if tn_path is None else str(tn_path),
            'image_tag': image_tag,
            'nir_dim': int(nir_dim),
            'target_mode': str(target_mode).strip().lower(),
            'image_channels': None if image_channels is None else int(image_channels),
            'wavelengths': None if wavelengths is None else [str(w) for w in wavelengths],
            'active_inputs': active_inputs_list,
            'required_files': required_files_list,
            'expected_sample_count': int(expected_sample_count),
            'ready_sample_count': int(ready_sample_count),
            'resolved_cache_mode': str(resolved_cache_mode),
            'is_complete': int(ready_sample_count) == int(expected_sample_count) and int(expected_sample_count) > 0,
            'created_at': created_at,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.save()


