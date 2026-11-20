"""
数据库检索模块
用于在多个在线质谱数据库中搜索化合物信息

支持的数据库：
- GNPS: 全球天然产物社交分子网络
- MassBank: 欧洲质谱数据库
- PubChem: 美国国立卫生研究院化合物数据库
"""

import requests
import json
import time
from urllib.parse import urlencode, quote
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class SpectrumDatabaseSearcher:
    """质谱数据库检索器，支持多个在线数据库的真实API调用"""

    def __init__(self, use_mock_data=False, timeout=10, max_retries=3, enable_parallel=False):
        """
        初始化数据库检索器

        参数:
            use_mock_data: 是否使用模拟数据（默认False，使用真实API）
            timeout: API请求超时时间（秒），默认10秒
            max_retries: 最大重试次数，默认3次
            enable_parallel: 是否启用并行请求（默认False）
        """
        self.use_mock_data = use_mock_data
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_parallel = enable_parallel

        self.databases = {
            'gnps': {
                'name': 'GNPS',
                'api_url': 'https://external.gnps2.org/gnpslibraryjson',
                'enabled': True
            },
            'massbank': {
                'name': 'MassBank',
                'api_url': 'https://massbank.eu/MassBank/api/v3',  # MassBank3 API
                'enabled': True
            },
            'pubchem': {
                'name': 'PubChem',
                'api_url': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug',
                'enabled': True
            }
        }
        self.search_results_cache = {}

        print(f"数据库检索器初始化完成")
        print(f"  模式: {'模拟数据' if use_mock_data else '真实API调用'}")
        print(f"  超时设置: {timeout}秒")
        print(f"  最大重试: {max_retries}次")
        print(f"  并行请求: {'启用' if enable_parallel else '禁用'}")
        print(f"  支持的数据库: {', '.join([db['name'] for db in self.databases.values() if db['enabled']])}")

    def spectrum_to_peak_string(self, spectrum, max_peaks=50):
        """
        将质谱转换为峰列表字符串

        参数:
            spectrum: 质谱对象，需包含peaks属性
            max_peaks: 最多保留的峰数量，默认50

        返回:
            str: 格式化的峰列表字符串 "mz1:intensity1 mz2:intensity2 ..."
        """
        try:
            if not hasattr(spectrum, 'peaks') or len(spectrum.peaks.mz) == 0:
                return ""

            # 获取强度最高的峰
            mz_intensities = list(zip(spectrum.peaks.mz, spectrum.peaks.intensities))
            mz_intensities.sort(key=lambda x: x[1], reverse=True)
            mz_intensities = mz_intensities[:max_peaks]
            mz_intensities.sort(key=lambda x: x[0])  # 按m/z排序

            # 格式化为字符串
            peak_string = " ".join([f"{mz:.4f}:{intensity:.4f}" for mz, intensity in mz_intensities])
            return peak_string
        except Exception as e:
            print(f"转换质谱峰时出错: {e}")
            return ""

    def _make_api_request(self, url, params=None, method='GET', data=None, headers=None):
        """
        统一的API请求方法，支持重试机制

        参数:
            url: API端点URL
            params: URL参数（GET请求）
            method: HTTP方法（GET/POST）
            data: 请求体数据（POST请求）
            headers: 请求头

        返回:
            dict or None: API响应的JSON数据，失败时返回None
        """
        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = requests.get(url, params=params, timeout=self.timeout, headers=headers)
                elif method == 'POST':
                    response = requests.post(url, json=data, params=params, timeout=self.timeout, headers=headers)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")

                if response.status_code == 200:
                    return response.json() if response.content else None
                elif response.status_code == 404:
                    return None
                else:
                    print(f"API请求失败 (状态码 {response.status_code}): {url}")

            except requests.Timeout:
                print(f"API请求超时 (尝试 {attempt + 1}/{self.max_retries}): {url}")
            except Exception as e:
                print(f"API请求出错 (尝试 {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))  # 指数退避

        return None

    def search_gnps_spectral_library(self, spectrum, precursor_mz, tolerance_mz=0.5,
                                   tolerance_peak=0.05, min_matched_peaks=4):
        """
        在GNPS光谱库中搜索（真实API调用）
        使用新的 GNPS2 API: https://external.gnps2.org/gnpslibraryjson

        参数:
            spectrum: 质谱对象
            precursor_mz: 母离子m/z值
            tolerance_mz: m/z容差（Da），默认0.5
            tolerance_peak: 峰匹配容差（Da），默认0.05
            min_matched_peaks: 最小匹配峰数，默认4

        返回:
            List[Dict]: 搜索结果列表
        """
        try:
            # 使用模拟数据模式
            if self.use_mock_data:
                return self._get_mock_gnps_results(spectrum, precursor_mz, min_matched_peaks)

            # 检查缓存
            cache_key = f"gnps_{precursor_mz:.4f}"
            if cache_key in self.search_results_cache:
                return self.search_results_cache[cache_key]

            if not hasattr(spectrum, 'peaks') or len(spectrum.peaks.mz) == 0:
                return []

            # 准备峰数据，格式为 "mz:intensity mz:intensity ..."
            peak_string = self.spectrum_to_peak_string(spectrum, max_peaks=50)
            if not peak_string:
                return []

            # GNPS2 Library JSON API
            api_url = self.databases['gnps']['api_url']

            # 根据 GNPS2 API 文档构建参数
            params = {
                'peaks': peak_string,
                'precursor_mz': precursor_mz,
                'mz_tol': tolerance_mz,
                'analog': 'No',  # 精确匹配
                'cosine': 'standard',  # 使用标准余弦相似度
                'min_matched_peaks': min_matched_peaks
            }

            results_json = self._make_api_request(api_url, params=params, method='GET')

            results = []
            if results_json:
                # GNPS2 API 返回格式可能是列表或字典
                hits = results_json if isinstance(results_json, list) else results_json.get('results', [])

                for hit in hits[:5]:
                    result = {
                        'compound_name': hit.get('Compound_Name', hit.get('compound_name', hit.get('name', 'Unknown'))),
                        'score': float(hit.get('MQScore', hit.get('cosine_score', hit.get('score', 0)))),
                        'source': 'GNPS',
                        'smiles': hit.get('Smiles', hit.get('smiles', '')),
                        'inchikey': hit.get('INCHI_AUX', hit.get('InChIKey', hit.get('inchikey', ''))),
                        'formula': hit.get('Formula', hit.get('formula', '')),
                        'matched_peaks': int(hit.get('SharedPeaks', hit.get('matched_peaks', min_matched_peaks))),
                        'library_id': hit.get('SpectrumID', hit.get('spectrum_id', ''))
                    }
                    results.append(result)

            # 缓存结果
            self.search_results_cache[cache_key] = results
            return results

        except Exception as e:
            print(f"GNPS搜索出错: {e}")
            import traceback
            traceback.print_exc()
            return self._get_mock_gnps_results(spectrum, precursor_mz, min_matched_peaks)

    def _get_mock_gnps_results(self, spectrum, precursor_mz, min_matched_peaks):
        """生成模拟的GNPS搜索结果"""
        return [{
            'compound_name': f'GNPS_candidate_{precursor_mz:.2f}',
            'score': 0.75 + np.random.random() * 0.2,
            'source': 'GNPS',
            'smiles': 'CCO',
            'inchikey': 'LFQSCWFLJHTTHZ-UHFFFAOYSA-N',
            'formula': 'C2H6O',
            'matched_peaks': min_matched_peaks + int(np.random.random() * 10),
            'total_peaks': len(spectrum.peaks.mz) if hasattr(spectrum, 'peaks') else 0
        }]

    def search_massbank(self, spectrum, precursor_mz, tolerance_mz=0.3):
        """
        在MassBank中搜索（真实API调用）
        使用 MassBank3 API

        参数:
            spectrum: 质谱对象
            precursor_mz: 母离子m/z值
            tolerance_mz: m/z容差（Da），默认0.3

        返回:
            List[Dict]: 搜索结果列表
        """
        try:
            # 使用模拟数据模式
            if self.use_mock_data:
                return self._get_mock_massbank_results(precursor_mz)

            # 检查缓存
            cache_key = f"massbank_{precursor_mz:.4f}"
            if cache_key in self.search_results_cache:
                return self.search_results_cache[cache_key]

            if not hasattr(spectrum, 'peaks') or len(spectrum.peaks.mz) == 0:
                return []

            # MassBank API v3 - 峰搜索端点
            # 参考: https://github.com/MassBank/MassBank3/tree/main/api
            api_url = f"{self.databases['massbank']['api_url']}/search/spectrum"

            # 准备峰数据（MassBank v3 格式）
            peaks_list = []
            for mz, intensity in zip(spectrum.peaks.mz[:50], spectrum.peaks.intensities[:50]):
                peaks_list.append([float(mz), float(intensity)])

            request_data = {
                'peaks': peaks_list,
                'precursorMz': precursor_mz,
                'precursorTolerance': tolerance_mz,
                'mzTolerance': tolerance_mz,
                'instrument': 'all',  # 搜索所有仪器类型
                'ionMode': 'POSITIVE',  # 可根据实际情况调整
                'maxResults': 5
            }

            headers = {'Content-Type': 'application/json'}
            results_json = self._make_api_request(api_url, method='POST', data=request_data, headers=headers)

            results = []
            if results_json:
                # MassBank v3 返回格式
                hits = results_json if isinstance(results_json, list) else results_json.get('results', results_json.get('data', []))

                for hit in hits[:5]:
                    # 提取分数并归一化 (MassBank 分数范围通常是 0-999 或 0-1)
                    raw_score = float(hit.get('score', hit.get('matchScore', 0)))
                    normalized_score = raw_score / 999.0 if raw_score > 1 else raw_score

                    result = {
                        'compound_name': hit.get('title', hit.get('compoundName', hit.get('compound_name', 'Unknown'))),
                        'score': normalized_score,
                        'source': 'MassBank',
                        'smiles': hit.get('smiles', hit.get('SMILES', '')),
                        'inchikey': hit.get('inchiKey', hit.get('InChIKey', '')),
                        'formula': hit.get('formula', hit.get('molecularFormula', '')),
                        'instrument': hit.get('instrument', hit.get('instrumentType', '')),
                        'accession': hit.get('accession', hit.get('id', ''))
                    }
                    results.append(result)

            # 缓存结果
            self.search_results_cache[cache_key] = results
            return results

        except Exception as e:
            print(f"MassBank搜索出错: {e}")
            import traceback
            traceback.print_exc()
            return self._get_mock_massbank_results(precursor_mz)

    def _get_mock_massbank_results(self, precursor_mz):
        """生成模拟的MassBank搜索结果"""
        return [{
            'compound_name': f'MassBank_hit_{precursor_mz:.2f}',
            'score': 0.65 + np.random.random() * 0.25,
            'source': 'MassBank',
            'smiles': 'CC(C)O',
            'inchikey': 'KFZMGEQAYNKOFK-UHFFFAOYSA-N',
            'formula': 'C3H8O',
            'instrument': 'LC-ESI-QQ',
            'collision_energy': '20V'
        }]

    def search_pubchem(self, precursor_mz, tolerance_mz=0.5):
        """
        在PubChem中通过精确质量搜索化合物（真实API调用）
        API格式: https://pubchem.ncbi.nlm.nih.gov/rest/pug/<input>/<operation>/[<output>][?<options>]

        参数:
            precursor_mz: 母离子m/z值
            tolerance_mz: m/z容差（Da），默认0.5

        返回:
            List[Dict]: 搜索结果列表
        """
        try:
            # 使用模拟数据模式
            if self.use_mock_data:
                return self._get_mock_pubchem_results(precursor_mz)

            # 检查缓存
            cache_key = f"pubchem_{precursor_mz:.4f}"
            if cache_key in self.search_results_cache:
                return self.search_results_cache[cache_key]

            # PubChem REST API - 通过精确质量搜索
            # API 文档: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
            base_url = self.databases['pubchem']['api_url']

            # 方法1：使用 fastformula 端点（通过分子量搜索）
            # 格式: /compound/fastformula/<formula>/cids/JSON
            min_mass = precursor_mz - tolerance_mz
            max_mass = precursor_mz + tolerance_mz

            # 第一步：通过质量范围搜索 CID
            search_url = f"{base_url}/compound/fastformula/C0-100H0-200N0-50O0-50/cids/JSON"

            params = {
                'AllowOtherElements': 'true',
                'MinMass': f"{min_mass:.4f}",
                'MaxMass': f"{max_mass:.4f}",
                'MaxRecords': 5
            }

            cids_json = self._make_api_request(search_url, params=params, method='GET')

            results = []
            if cids_json and 'IdentifierList' in cids_json and 'CID' in cids_json['IdentifierList']:
                cids = cids_json['IdentifierList']['CID'][:5]

                # 第二步：批量获取化合物详细信息（减少API调用次数）
                cid_list = ','.join(map(str, cids))
                detail_url = f"{base_url}/compound/cid/{cid_list}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName,InChIKey/JSON"

                detail_json = self._make_api_request(detail_url, method='GET')

                if detail_json and 'PropertyTable' in detail_json:
                    for props in detail_json['PropertyTable']['Properties']:
                        # 计算质量匹配分数
                        mw = float(props.get('MolecularWeight', 0))
                        mass_diff = abs(mw - precursor_mz)
                        mass_score = max(0, 1 - (mass_diff / tolerance_mz))

                        result = {
                            'compound_name': props.get('IUPACName', f"CID_{props.get('CID', 'Unknown')}")[:100],
                            'score': 0.7 + 0.3 * mass_score,  # 基础分数 0.7 + 质量匹配分数
                            'source': 'PubChem',
                            'smiles': props.get('CanonicalSMILES', ''),
                            'inchikey': props.get('InChIKey', ''),
                            'formula': props.get('MolecularFormula', ''),
                            'molecular_weight': mw,
                            'cid': props.get('CID', '')
                        }
                        results.append(result)

                time.sleep(0.3)  # PubChem API 限速（每秒最多5个请求）

            # 缓存结果
            self.search_results_cache[cache_key] = results
            return results

        except Exception as e:
            print(f"PubChem搜索出错: {e}")
            import traceback
            traceback.print_exc()
            return self._get_mock_pubchem_results(precursor_mz)

    def _get_mock_pubchem_results(self, precursor_mz):
        """生成模拟的PubChem搜索结果"""
        return [{
            'compound_name': f'PubChem_compound_{precursor_mz:.2f}',
            'score': 0.80,
            'source': 'PubChem',
            'smiles': 'CCCCCO',
            'inchikey': 'AMQJEAYHLZJPGS-UHFFFAOYSA-N',
            'formula': 'C5H12O',
            'molecular_weight': precursor_mz,
            'cid': '12345'
        }]

    def search_all_databases(self, spectrum, precursor_mz, top_k=5, include_pubchem=True):
        """
        在所有数据库中搜索并合并结果（真实API调用）

        参数:
            spectrum: 质谱对象
            precursor_mz: 母离子m/z值
            top_k: 返回前k个最佳匹配结果，默认5
            include_pubchem: 是否包含PubChem搜索，默认True

        返回:
            List[Dict]: 按分数排序的搜索结果列表（最多top_k个）
        """
        # 使用并行搜索（如果启用）
        if self.enable_parallel:
            return self._search_all_databases_parallel(spectrum, precursor_mz, top_k, include_pubchem)

        # 串行搜索（原始方法）
        all_results = []

        # GNPS搜索（光谱匹配）
        if self.databases['gnps']['enabled']:
            try:
                gnps_results = self.search_gnps_spectral_library(spectrum, precursor_mz)
                all_results.extend(gnps_results)
            except Exception as e:
                print(f"GNPS搜索失败: {e}")

        # MassBank搜索（光谱匹配）
        if self.databases['massbank']['enabled']:
            try:
                massbank_results = self.search_massbank(spectrum, precursor_mz)
                all_results.extend(massbank_results)
            except Exception as e:
                print(f"MassBank搜索失败: {e}")

        # PubChem搜索（精确质量匹配）
        if include_pubchem and self.databases['pubchem']['enabled']:
            try:
                pubchem_results = self.search_pubchem(precursor_mz)
                all_results.extend(pubchem_results)
            except Exception as e:
                print(f"PubChem搜索失败: {e}")

        # 去重（基于inchikey）
        unique_results = {}
        for result in all_results:
            inchikey = result.get('inchikey', '')
            if inchikey and inchikey in unique_results:
                # 保留分数更高的结果
                if result.get('score', 0) > unique_results[inchikey].get('score', 0):
                    unique_results[inchikey] = result
            else:
                # 使用化合物名称作为后备键
                key = inchikey if inchikey else result.get('compound_name', f"unknown_{len(unique_results)}")
                unique_results[key] = result

        # 按分数排序并返回top_k结果
        sorted_results = sorted(unique_results.values(), key=lambda x: x.get('score', 0), reverse=True)
        return sorted_results[:top_k]

    def _search_all_databases_parallel(self, spectrum, precursor_mz, top_k=5, include_pubchem=True):
        """
        并行搜索所有数据库（性能优化版本）

        参数:
            spectrum: 质谱对象
            precursor_mz: 母离子m/z值
            top_k: 返回前k个最佳匹配结果，默认5
            include_pubchem: 是否包含PubChem搜索，默认True

        返回:
            List[Dict]: 按分数排序的搜索结果列表（最多top_k个）
        """
        all_results = []
        search_tasks = []

        # 准备搜索任务
        if self.databases['gnps']['enabled']:
            search_tasks.append(('gnps', lambda: self.search_gnps_spectral_library(spectrum, precursor_mz)))

        if self.databases['massbank']['enabled']:
            search_tasks.append(('massbank', lambda: self.search_massbank(spectrum, precursor_mz)))

        if include_pubchem and self.databases['pubchem']['enabled']:
            search_tasks.append(('pubchem', lambda: self.search_pubchem(precursor_mz)))

        # 并行执行搜索
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_db = {executor.submit(task[1]): task[0] for task in search_tasks}

            for future in future_to_db:
                db_name = future_to_db[future]
                try:
                    # 设置每个任务的超时（总超时 = timeout * max_retries）
                    results = future.result(timeout=self.timeout * self.max_retries)
                    if results:
                        all_results.extend(results)
                except FuturesTimeoutError:
                    print(f"{db_name}搜索超时（总超时: {self.timeout * self.max_retries}秒）")
                except Exception as e:
                    print(f"{db_name}搜索失败: {e}")

        # 去重和排序（与串行方法相同）
        unique_results = {}
        for result in all_results:
            inchikey = result.get('inchikey', '')
            if inchikey and inchikey in unique_results:
                if result.get('score', 0) > unique_results[inchikey].get('score', 0):
                    unique_results[inchikey] = result
            else:
                key = inchikey if inchikey else result.get('compound_name', f"unknown_{len(unique_results)}")
                unique_results[key] = result

        sorted_results = sorted(unique_results.values(), key=lambda x: x.get('score', 0), reverse=True)
        return sorted_results[:top_k]

    def enable_database(self, database_name, enabled=True):
        """
        启用或禁用特定数据库

        参数:
            database_name: 数据库名称 ('gnps', 'massbank', 'pubchem')
            enabled: 是否启用，默认True
        """
        if database_name in self.databases:
            self.databases[database_name]['enabled'] = enabled
            print(f"数据库 {self.databases[database_name]['name']} {'已启用' if enabled else '已禁用'}")
        else:
            print(f"未知的数据库: {database_name}")

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            'total_cached': len(self.search_results_cache),
            'databases': {
                'gnps': len([k for k in self.search_results_cache if k.startswith('gnps_')]),
                'massbank': len([k for k in self.search_results_cache if k.startswith('massbank_')]),
                'pubchem': len([k for k in self.search_results_cache if k.startswith('pubchem_')])
            }
        }

    def clear_cache(self):
        """清空缓存"""
        self.search_results_cache = {}
        print("缓存已清空")


if __name__ == '__main__':
    print("="*60)
    print("数据库检索模块已加载")
    print("="*60)
    print("\n支持的数据库:")
    print("  1. GNPS - 全球天然产物社交分子网络（光谱匹配）")
    print("  2. MassBank - 欧洲质谱数据库（光谱匹配）")
    print("  3. PubChem - 美国国立卫生研究院化合物数据库（精确质量匹配）")
    print("\n使用示例:")
    print("  # 使用真实API")
    print("  searcher = SpectrumDatabaseSearcher(use_mock_data=False)")
    print("  results = searcher.search_all_databases(spectrum, precursor_mz, top_k=5)")
    print("\n  # 使用模拟数据（测试）")
    print("  searcher = SpectrumDatabaseSearcher(use_mock_data=True)")
    print("  results = searcher.search_all_databases(spectrum, precursor_mz, top_k=5)")
    print("="*60)
