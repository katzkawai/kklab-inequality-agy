# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///
"""
Zenodo REST API を利用して論文・コード・データを自動登録し、DOI を発行・取得するスクリプト
PEP 723 (Inline Script Metadata) 準拠
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import requests

ZENODO_API_URL = "https://zenodo.org/api"


def main():
    token = os.environ.get("ZENODO_API_KEY")
    if not token:
        print("❌ エラー: 環境変数 ZENODO_API_KEY が設定されていません。")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}

    print("=" * 70)
    print("  Zenodo への論文・ソースコード・データの自動登録とDOI発行")
    print("=" * 70)

    # 1. 新規デポジットの作成
    print("\n[Step 1] Zenodo に新規デポジット（下書きレコード）を作成中...")
    r = requests.post(
        f"{ZENODO_API_URL}/deposit/depositions",
        params={"access_token": token},
        json={},
        headers={"Content-Type": "application/json"}
    )
    if r.status_code not in (200, 201):
        print(f"❌ デポジット作成エラー ({r.status_code}):", r.text)
        sys.exit(1)

    deposition = r.json()
    dep_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    html_url = deposition["links"]["html"]
    print(f"  -> デポジット作成成功! ID: {dep_id}")
    print(f"  -> 管理URL: {html_url}")

    # 2. ファイルのアップロード
    files_to_upload = [
        ("docs/theil_paper.pdf", "theil_paper.pdf"),
        ("paper/theil_paper.tex", "theil_paper.tex"),
        ("calculate_theil.py", "calculate_theil.py"),
        ("output/household_micro_data.csv", "household_micro_data.csv"),
        ("output/theil_decomposition_summary.csv", "theil_decomposition_summary.csv"),
        ("docs/theil_decomposition.png", "theil_decomposition.png"),
        ("CITATION.cff", "CITATION.cff"),
        ("README.md", "README.md"),
        ("LICENSE", "LICENSE"),
    ]

    print(f"\n[Step 2] {len(files_to_upload)} 個の成果物ファイルをアップロード中...")
    for local_path, target_name in files_to_upload:
        p = Path(local_path)
        if not p.exists():
            print(f"  ⚠️ スキップ（ファイル未検出）: {local_path}")
            continue

        print(f"  -> アップロード中: {target_name} ({p.stat().st_size:,} bytes)...")
        with open(p, "rb") as fp:
            upload_r = requests.put(
                f"{bucket_url}/{target_name}",
                data=fp,
                params={"access_token": token},
            )
            if upload_r.status_code not in (200, 201):
                print(f"    ❌ アップロード失敗 ({upload_r.status_code}):", upload_r.text)
                sys.exit(1)
            print(f"    ✅ 完了: {target_name}")

    # 3. メタデータの設定
    print("\n[Step 3] メタデータ（書誌情報）を設定中...")
    metadata = {
        "metadata": {
            "title": "日本の家計所得格差におけるタイテル指数の加法分解とデータ生成・計算・視覚化手法の計算論的実装",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": (
                "<p>本研究は、日本の所得格差の構造的要因を解明するため、一般化エントロピー指数族である"
                "<strong>タイテル指数（Theil-T Index, GE(1)）</strong>の完全加法分解（グループ内格差・グループ間格差）を"
                "厳密に計算・検証する計算論的フレームワークおよび実証解説論文を提供します。</p>"
                "<h4>内容物</h4>"
                "<ul>"
                "<li><strong>theil_paper.pdf</strong>: 日本語LuaLaTeX（jlreq）による学術解説論文</li>"
                "<li><strong>theil_paper.tex</strong>: 論文LaTeXソースファイル</li>"
                "<li><strong>calculate_theil.py</strong>: PEP 723準拠の完全なPython実行スクリプト</li>"
                "<li><strong>household_micro_data.csv</strong>: 日本の所得分布を模した合成ミクロデータ（1,500世帯）</li>"
                "<li><strong>theil_decomposition_summary.csv</strong>: 年齢階層別の加法分解集計データ</li>"
                "<li><strong>theil_decomposition.png</strong>: 4面統合可視化ダッシュボード画像</li>"
                "</ul>"
                "<p>Webレポート: <a href='http://katzkawai.org/kklab-inequality-agy/'>http://katzkawai.org/kklab-inequality-agy/</a><br>"
                "GitHub: <a href='https://github.com/katzkawai/kklab-inequality-agy'>https://github.com/katzkawai/kklab-inequality-agy</a></p>"
            ),
            "creators": [
                {
                    "name": "河合 勝彦",
                    "affiliation": "名古屋市立大学大学院経済学研究科"
                }
            ],
            "access_right": "open",
            "license": "MIT",
            "keywords": [
                "タイテル指数",
                "Theil Index",
                "Theil-T Index",
                "所得格差",
                "加法分解",
                "Additive Decomposition",
                "一般化エントロピー指数",
                "国民生活基礎調査",
                "家計調査",
                "Python",
                "PEP 723",
                "LuaLaTeX",
                "jlreq",
                "オープンサイエンス"
            ],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/katzkawai/kklab-inequality-agy",
                    "relation": "isSupplementTo",
                    "scheme": "url"
                },
                {
                    "identifier": "http://katzkawai.org/kklab-inequality-agy/",
                    "relation": "isDocumentedBy",
                    "scheme": "url"
                }
            ]
        }
    }

    meta_r = requests.put(
        f"{ZENODO_API_URL}/deposit/depositions/{dep_id}",
        params={"access_token": token},
        data=json.dumps(metadata),
        headers={"Content-Type": "application/json"}
    )
    if meta_r.status_code != 200:
        print(f"❌ メタデータ設定エラー ({meta_r.status_code}):", meta_r.text)
        sys.exit(1)
    print("  ✅ メタデータ設定完了!")

    # 4. 公開 (Publish) の実行
    print("\n[Step 4] レコードを公開 (Publish) して DOI を発行中...")
    pub_r = requests.post(
        f"{ZENODO_API_URL}/deposit/depositions/{dep_id}/actions/publish",
        params={"access_token": token}
    )
    if pub_r.status_code != 202 and pub_r.status_code != 200:
        print(f"❌ 公開エラー ({pub_r.status_code}):", pub_r.text)
        sys.exit(1)

    result = pub_r.json()
    doi = result.get("doi")
    doi_url = result.get("doi_url")
    record_id = result.get("record_id")
    record_url = result.get("links", {}).get("html", f"https://zenodo.org/records/{record_id}")

    print("\n" + "=" * 70)
    print("🎉 Zenodo への登録および公開が完了しました！")
    print("=" * 70)
    print(f"  ● 発行された DOI: {doi}")
    print(f"  ● DOI URL       : {doi_url}")
    print(f"  ● Zenodo レコード: {record_url}")
    print("=" * 70)

    # 結果を json に保存
    zenodo_result = {
        "deposition_id": dep_id,
        "record_id": record_id,
        "doi": doi,
        "doi_url": doi_url,
        "record_url": record_url
    }
    with open("zenodo_registration.json", "w", encoding="utf-8") as f:
        json.dump(zenodo_result, f, ensure_ascii=False, indent=2)
    print("✅ 登録結果を zenodo_registration.json に保存しました。")


if __name__ == "__main__":
    main()
