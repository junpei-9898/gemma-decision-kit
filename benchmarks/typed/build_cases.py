# SPDX-License-Identifier: Apache-2.0
"""Create fictional, precommitted Japanese typed-decision cases; no predictions used."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).parent

def main():
    rows=[];gold={}
    def add(cid,group,kind,state,instructions,criteria,expected,reason):
        body={'state':state,'questions':{'q':{'type':kind,'instructions':instructions,'criteria':criteria}}}
        rows.append({'id':cid,'group':group,'body':body})
        gold[cid]={'expected':expected,'reason':reason,'status':'AI_PROVISIONAL_NOT_HUMAN_GOLD'}
    binary=[
      ('荷物は本日、受取人が受領した。','荷物は配達済みです。',True),
      ('荷物は未発送で倉庫にある。','荷物は配達済みです。',False),
      ('受付は17時まで。申請は16時に到着した。','申請は締切に間に合いました。',True),
      ('受付は17時まで。申請は18時に到着した。','申請は締切に間に合いました。',False),
      ('旧会場は西館。確定した最新版では東館に変更され、後の変更はない。','現在の会場は東館です。',True),
      ('旧会場は西館。確定した最新版では東館に変更され、後の変更はない。','現在の会場は西館です。',False),
      ('商品は単価120円で8個。合計は単価と個数の積で、追加料金はない。','合計は960円です。',True),
      ('商品は単価120円で8個。合計は単価と個数の積で、追加料金はない。','合計は1080円です。',False),
      ('利用は社内だけに限定し、外部公開を禁止した。','外部への公開は禁止されています。',True),
      ('利用は社内だけに限定し、外部公開を禁止した。','外部への公開が許可されています。',False),
      ('受付条件は身分証と申込書の両方。二つとも提出済みである。','受付条件を満たしています。',True),
      ('受付条件は身分証と申込書の両方。申込書だけ提出した。','受付条件を満たしています。',False),
      ('会議は中止されていない。予定どおり開催すると正式通知された。','会議は開催予定です。',True),
      ('会議は正式に中止された。延期や再開の予定はない。','会議は開催予定です。',False),
      ('担当は田中、承認者は佐藤と記録されている。','佐藤は承認者です。',True),
      ('担当は田中、承認者は佐藤と記録されている。','田中は承認者です。',False),
      ('在庫は5個。3個出荷し、他の入出庫はない。','在庫は2個です。',True),
      ('在庫は5個。3個出荷し、他の入出庫はない。','在庫は3個です。',False),
      ('無償交換の条件は購入後30日以内。今回は購入後20日である。','今回は無償交換の期間内です。',True),
      ('無償交換の条件は購入後30日以内。今回は購入後40日である。','今回は無償交換の期間内です。',False),
    ]
    for i,(state,claim,expected) in enumerate(binary):
        ins='記録だけで次の主張の真偽を判定：'+claim
        criteria={'true':'主張は正しい','false':'主張は誤り'}
        add(f'bool-{i:02d}',f'binary-{i//2}','noul',state,ins,criteria,expected,'明記された状態・条件・算術に照らした真偽。')
        add(f'choice2-{i:02d}',f'binary-{i//2}','choice',state,ins,criteria,str(expected).lower(),'同じ二値命題のchoice形式。')
    departments={'billing':'請求・返金','technical':'故障・不具合','sales':'購入・見積','account':'ログイン・退会'}
    texts=[
      ['二重請求されたので返金してほしい。','支払った金額と請求書の金額が違う。','返品は受理済み。返金の進捗を確認したい。','ログインはできるが、請求の内訳に質問がある。','見積は不要で、誤請求の訂正だけを求めている。'],
      ['機器が起動しない。故障の調査をお願いしたい。','画面が突然消える不具合が続いている。','更新後にアプリが必ず異常終了する。','支払いは完了しているが、機器が故障した。','アカウントは正常で、通信機器の不具合を相談したい。'],
      ['新規購入を検討しているので見積がほしい。','追加で10台購入する場合の価格を知りたい。','導入前に契約プランを相談したい。','故障対応ではなく、新しい機種の購入相談である。','請求済みの話ではなく、来月の導入費用を見積もってほしい。'],
      ['パスワードを忘れてログインできない。','アカウントを退会したい。','二要素認証の設定を変更したい。','支払いに問題はなく、ログイン名の変更を希望する。','製品は正常だが、登録アカウントを削除したい。']]
    for cat,states in zip(departments,texts):
        for j,state in enumerate(states):
            add(f'choice4-{cat}-{j}',f'route4-{cat}','choice',state,'問い合わせの主目的を担当する部署を一つ選ぶ。',departments,cat,'主目的に対応する部署。')
    kinds={'refund':'返金','repair':'修理','quote':'見積','password':'パスワード','cancel':'解約','shipping':'配送','address':'住所変更','privacy':'個人情報開示'}
    texts8=[('返金をお願いします。','修理は不要で返金だけを希望します。'),('修理をお願いします。','返金ではなく修理を希望します。'),('購入前の見積がほしい。','まだ購入しておらず見積だけが必要です。'),('パスワードを再設定したい。','契約は続けるがパスワードを忘れました。'),('契約を解約したい。','パスワード変更ではなく解約を希望します。'),('配送の到着予定を確認したい。','住所は正しいので配送の遅延状況を教えてください。'),('登録住所を変更したい。','配送状況ではなく登録住所の更新をお願いします。'),('保有する私の個人情報を開示してほしい。','住所変更ではなく個人情報の開示を請求します。')]
    for cat,states in zip(kinds,texts8):
        for j,state in enumerate(states):add(f'choice8-{cat}-{j}',f'route8-{cat}','choice',state,'依頼の主目的を一つ選ぶ。',kinds,cat,'明示された依頼の主目的。')
    rubric_sets=[
      ('checklist',['完了0項目','完了1項目','完了2項目','完了3項目','完了4項目'],['A未完了、B未完了、C未完了、D未完了。','A完了、B未完了、C未完了、D未完了。','A完了、B未完了、C完了、D未完了。','A未完了、B完了、C完了、D完了。','A完了、B完了、C完了、D完了。'],'4項目の完了数で評価する。'),
      ('delay',['遅延なし','遅延1時間','遅延2時間','遅延3時間','遅延4時間'],['予定と実際の到着はいずれも10時。','予定10時、実際11時に到着。','予定10時、実際12時に到着。','予定10時、実際13時に到着。','予定10時、実際14時に到着。'],'同日の予定時刻からの遅延時間で評価する。'),
      ('satisfaction',['非常に不満','やや不満','中立','やや満足','非常に満足'],['非常に不満です。全く期待に届きません。','やや不満です。少し改善してほしいです。','満足でも不満でもありません。','やや満足です。概ね期待どおりです。','非常に満足です。期待を大きく上回りました。'],'書き手が表明した満足度を評価する。'),
      ('impact',['影響なし','1人に影響','2人に影響','3人に影響','4人に影響'],['4人とも正常で、影響はない。','Aのみ障害、BとCとDは正常。','AとBに障害、CとDは正常。','AとBとCに障害、Dは正常。','AとBとCとDの全員に障害。'],'影響を受けた人数で評価する。')]
    for family,criteria,states,ins in rubric_sets:
        for j,state in enumerate(states):add(f'score-{family}-{j}',f'score-{family}','score',state,ins,criteria,j,'事前定義した順序尺度の該当位置。')
    (ROOT/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'gold.json').write_text(json.dumps(gold,ensure_ascii=False,indent=2)+'\n')
    manifest={'case_count':len(rows),'groups':len({x['group'] for x in rows}),'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['cases.json','gold.json']},'label_scope':'AI provisional; generated before all model predictions; binary/choice2 share groups; category variants correlated'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
if __name__=='__main__':main()
