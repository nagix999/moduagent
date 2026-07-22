---
name: invoice-review
description: 회사 정책에 따라 청구서를 검토한다. 청구 금액, 증빙, 승인 여부를 확인할 때 사용한다.
license: MIT
compatibility: ModuAgent 0.2.0
metadata:
  version: "1.0.0"
allowed-tools: lookup_invoice
---

# Invoice review

1. 사용자에게서 청구서 ID를 확인한다.
2. `lookup_invoice`로 원본 금액, 증빙 첨부 여부, 승인 상태를 조회한다.
3. `references/policy.md`를 읽고 금액 구간에 맞는 승인 조건을 적용한다.
4. 사실과 정책 근거를 함께 제시하고, 조회되지 않은 내용은 추측하지 않는다.
5. 최종 결과는 `assets/report-template.md`의 항목 순서로 간결하게 작성한다.

Reference나 asset을 읽을 때는 필요한 파일만 한 번씩 읽는다. 정보가 부족하면 승인으로 간주하지 말고 `확인 필요`로 표시한다.
