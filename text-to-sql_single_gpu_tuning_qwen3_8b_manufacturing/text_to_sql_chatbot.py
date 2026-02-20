# -*- coding: utf-8 -*-
import re
import gradio as gr
import pymysql
from openai import OpenAI

# ============== 설정 ==============
RUNPOD_ENDPOINT_ID = '0mqamq4vza8t9g'
API_KEY = '런팟 키 값'
MODEL_NAME = "iamjoon/qwen3-14b-text-to-sql-ko-checkpoint-700"

# MariaDB 연결 설정
DB_CONFIG = {
    'host': 'localhost',
    'port': 13306,
    'user': 'master',
    'password': 'master1234~',
    'database': 'mesdb',
    'charset': 'utf8mb4'
}

# OpenAI 클라이언트
client = OpenAI(
    api_key=API_KEY,
    base_url=f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
)

# 시스템 프롬프트
SYSTEM_PROMPT = '''당신은 SQL을 생성하는 AI 모델입니다.
아래는 데이터베이스 스키마(DDL)입니다.

<SCHEMA>
-- 작업완료로그 테이블
CREATE TABLE IF NOT EXISTS `log_lottranslog` (
  `TRANSLOGID` bigint(20) NOT NULL DEFAULT 0 COMMENT '로그ID',
  `LOTNO` char(20) NOT NULL DEFAULT '' COMMENT 'LOT번호',
  `LINENO` char(20) DEFAULT NULL COMMENT '생산라인번호',
  `TRANSACTIONNAME` char(50) NOT NULL DEFAULT '' COMMENT '처리명',
  `TIMELOGGED` datetime(3) NOT NULL DEFAULT '0000-00-00 00:00:00.000' COMMENT '로그입력일시',
  `ACTUALTIME` datetime(3) NOT NULL DEFAULT '0000-00-00 00:00:00.000' COMMENT '실제실행일시',
  `MATERIALCODE` char(30) NOT NULL DEFAULT '' COMMENT '자재코드',
  `MATERIALNAME` char(50) NOT NULL DEFAULT '' COMMENT '자재명',
  `TRANSQTY` double NOT NULL DEFAULT 0 COMMENT '변경수량',
  `CURRENTQTY` double NOT NULL DEFAULT 0 COMMENT '현재수량',
  `NEXTQTY` double DEFAULT NULL COMMENT '변경반영된수량',
  `TRANSUOM` char(5) DEFAULT NULL COMMENT '측정단위',
  `WAREHOUSECODE` char(20) DEFAULT NULL COMMENT '창고코드',
  `BOPMATERIALCODE` char(30) DEFAULT NULL COMMENT '자재코드',
  `PROCESSCODE` char(20) DEFAULT NULL COMMENT '공정코드',
  `USERCODE` char(20) NOT NULL DEFAULT '' COMMENT '사용자코드',
  PRIMARY KEY (`TRANSLOGID`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci COMMENT='LOT 변경 로그';

-- VIEW 종속성 오류를 극복하기 위해 임시 테이블을 생성합니다.
CREATE TABLE `vw_lot` (
    `lotno` CHAR(20) NOT NULL COLLATE 'utf8_general_ci' COMMENT 'LOT번호',
    `lineno` CHAR(20) NOT NULL COLLATE 'utf8_general_ci' COMMENT '생산라인번호',
    `materialcode` CHAR(30) NOT NULL COLLATE 'utf8_general_ci' COMMENT '자재코드',
    `materialname` CHAR(50) NOT NULL COLLATE 'utf8_general_ci' COMMENT '자재명',
    `customercode` CHAR(20) NULL COLLATE 'utf8_general_ci' COMMENT '고객코드',
    `comcode2` CHAR(20) NULL COLLATE 'utf8_general_ci' COMMENT '',
    `createdate` DATETIME(3) NOT NULL COMMENT '생성일시',
    `duedate` DATETIME(3) NULL COMMENT '마감일시',
    `createqty` DOUBLE NOT NULL COMMENT '생성수량',
    `standarduom` CHAR(5) NULL COLLATE 'utf8_general_ci' COMMENT '표준측정단위',
    `ispr` CHAR(1) NOT NULL COLLATE 'utf8_general_ci' COMMENT '',
    `code1` CHAR(50) NULL COLLATE 'utf8_general_ci' COMMENT '',
    `code2` CHAR(50) NULL COLLATE 'utf8_general_ci' COMMENT '',
    `order_id` CHAR(20) NULL COLLATE 'utf8_general_ci' COMMENT '주문ID',
    `confirmqty` CHAR(1) NOT NULL COLLATE 'utf8_general_ci' COMMENT '수량확인여부',
    `lotindex` SMALLINT(6) NOT NULL COMMENT 'LOT인덱스',
    `warehousecode` CHAR(20) NULL COLLATE 'utf8_general_ci' COMMENT '창고코드',
    `bopmaterialcode` CHAR(30) NULL COLLATE 'utf8_general_ci' COMMENT '자재코드',
    `processcode` CHAR(20) NULL COLLATE 'utf8_general_ci' COMMENT '공정코드',
    `workstate` CHAR(10) NOT NULL COLLATE 'utf8_general_ci' COMMENT '진행상태구분(QUEUED, PROCESSING, FINISHED, FAILED, HOLD)',
    `queuedqty` DOUBLE NULL COMMENT '대기수량',
    `startedqty` DOUBLE NULL COMMENT '시작수량',
    `queuedtime` DATETIME(3) NULL COMMENT '대기일시',
    `startedtime` DATETIME(3) NULL COMMENT '시작일시',
    `consumptiondone` CHAR(1) NOT NULL COLLATE 'utf8_general_ci' COMMENT '소모완료여부',
    `datacollectdone` CHAR(1) NOT NULL COLLATE 'utf8_general_ci' COMMENT '데이터수집완료여부',
    `completedone` CHAR(1) NOT NULL COLLATE 'utf8_general_ci' COMMENT '완료여부'
) ENGINE=MyISAM;

-- 자재정보 테이블
CREATE TABLE IF NOT EXISTS `udt_material` (
  `MATERIALCODE` char(30) NOT NULL DEFAULT '' COMMENT '자재코드',
  `MATERIALNAME` char(50) NOT NULL DEFAULT '' COMMENT '자재명',
  `MATERIALTYPECODE` char(20) NOT NULL DEFAULT '' COMMENT '자재구분코드',
  `LOTQTY` double DEFAULT NULL COMMENT 'LOT수량',
  `STANDARDUOM` char(5) DEFAULT NULL COMMENT '표준측정단위',
  `DESCRIPTION` char(100) DEFAULT NULL COMMENT '설명',
  `REMARK` char(50) DEFAULT NULL COMMENT '비고',
  PRIMARY KEY (`MATERIALCODE`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- LOT 변경 로그 테이블
CREATE TABLE IF NOT EXISTS `log_completelog` (
  `TRANSLOGID` bigint(20) NOT NULL DEFAULT 0 COMMENT '트랜잭션아이디',
  `QUEUETIME` double NOT NULL DEFAULT 0 COMMENT '대기시간',
  `PROCESSTIME` double NOT NULL DEFAULT 0 COMMENT '처리시간',
  `NEXTBOPMATERIALCODE` char(30) NOT NULL DEFAULT '' COMMENT '다음공정자재코드',
  `NEXTPROCESSCODE` char(20) NOT NULL DEFAULT '' COMMENT '다음공정코드',
  `MACHINECODE` char(20) DEFAULT NULL COMMENT '',
  PRIMARY KEY (`TRANSLOGID`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- 공정완료로그 테이블
CREATE TABLE IF NOT EXISTS `log_finishlog` (
  `TRANSLOGID` bigint(20) NOT NULL DEFAULT 0 COMMENT '로그ID',
  `QUEUETIME` double NOT NULL DEFAULT 0 COMMENT '대기시간',
  `PROCESSTIME` double NOT NULL DEFAULT 0 COMMENT '처리시간',
  `NEXTWAREHOUSECODE` char(20) NOT NULL DEFAULT '' COMMENT '이동된창고코드',
  `MACHINECODE` char(20) DEFAULT NULL COMMENT '장비코드',
  PRIMARY KEY (`TRANSLOGID`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- LOT 공정 진행상황 정보 테이블
CREATE TABLE IF NOT EXISTS `sgt_wiplot` (
  `LOTNO` char(20) NOT NULL COMMENT 'LOT번호',
  `LINENO` char(20) NOT NULL COMMENT '생산라인번호',
  `LOTINDEX` smallint(6) NOT NULL DEFAULT 0 COMMENT 'LOT인덱스',
  `WAREHOUSECODE` char(20) DEFAULT NULL COMMENT '창고코드',
  `BOPMATERIALCODE` char(30) DEFAULT NULL COMMENT '자재코드',
  `PROCESSCODE` char(20) DEFAULT NULL COMMENT '공정코드',
  `WORKSTATE` char(10) NOT NULL DEFAULT '' COMMENT '진행상태구분(QUEUED, PROCESSING, FINISHED, FAILED, HOLD)',
  `QUEUEDQTY` double DEFAULT NULL COMMENT '입력수량',
  `STARTEDQTY` double DEFAULT NULL COMMENT '작업수량',
  `QUEUEDTIME` datetime(3) DEFAULT NULL COMMENT '대기일시',
  `STARTEDTIME` datetime(3) DEFAULT NULL COMMENT '시작일시',
  `CONSUMPTIONDONE` char(1) NOT NULL DEFAULT '' COMMENT '소모완료여부',
  `DATACOLLECTDONE` char(1) NOT NULL DEFAULT '' COMMENT '데이터수집완료여부',
  `COMPLETEDONE` char(1) NOT NULL DEFAULT '' COMMENT '완료여부',
  PRIMARY KEY (`LOTINDEX`,`LOTNO`,`LINENO`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- 공장생산라인정보 테이블
CREATE TABLE IF NOT EXISTS `udt_factoryline` (
  `LINENO` char(20) NOT NULL COMMENT '생산라인번호',
  `DESCRIPTION` char(100) DEFAULT NULL COMMENT '설명',
  `FACTORYCODE` char(20) NOT NULL COMMENT '공장코드',
  `MESYN` int(11) NOT NULL DEFAULT 0 COMMENT '자동화여부',
  `AUTOINPUTMATYN` int(11) DEFAULT NULL COMMENT '자재자동공급여부',
  PRIMARY KEY (`LINENO`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- 공정정보 테이블
CREATE TABLE IF NOT EXISTS `udt_process` (
  `PROCESSCODE` char(20) NOT NULL DEFAULT '' COMMENT '공정코드',
  `PROCESSNAME` char(50) NOT NULL DEFAULT '' COMMENT '공정명',
  `COMPLETEONLY` char(1) NOT NULL DEFAULT '' COMMENT '',
  `LOTFLOW` char(1) NOT NULL DEFAULT '' COMMENT '',
  `WORKSTATIONCODE` char(20) DEFAULT NULL COMMENT '워크스테이션코드',
  `DESCRIPTION` char(100) DEFAULT NULL COMMENT '설명',
  PRIMARY KEY (`PROCESSCODE`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
</SCHEMA>

규칙:
1. 반드시 스키마에 존재하는 테이블과 컬럼만 사용하십시오.
2. 답변은 SQL 쿼리만 출력하고 설명은 절대 포함하지 마십시오.
3. WHERE / GROUP BY / ORDER BY는 스키마 기반으로 논리적으로 작성하십시오.
4. 쿼리는 항상 실행 가능한 SQL이어야 합니다.'''


def generate_sql(user_question: str) -> str:
    """자연어 질문을 SQL로 변환"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0
    )
    
    raw = response.choices[0].message.content
    # <think>...</think> 제거
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    sql = clean.strip()
    
    return sql


def execute_sql(sql: str):
    """SQL 실행 후 결과 반환"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
        return columns, rows, None
    except Exception as e:
        return None, None, str(e)
    finally:
        conn.close()


def chat(user_message: str, history: list):
    """챗봇 메인 함수"""
    if not user_message.strip():
        return history, ""
    
    # 1. SQL 생성
    try:
        sql = generate_sql(user_message)
    except Exception as e:
        history.append((user_message, f"❌ SQL 생성 오류: {str(e)}"))
        return history, ""
    
    # 2. SQL 실행
    columns, rows, error = execute_sql(sql)
    
    # 3. 결과 포맷팅
    if error:
        response = f"**생성된 SQL:**\n```sql\n{sql}\n```\n\n❌ **실행 오류:** {error}"
    elif not rows:
        response = f"**생성된 SQL:**\n```sql\n{sql}\n```\n\n📭 **결과:** 데이터 없음"
    else:
        # 테이블 형태로 결과 표시
        table_header = " | ".join(columns)
        table_sep = " | ".join(["---"] * len(columns))
        table_rows = "\n".join([" | ".join(str(cell) for cell in row) for row in rows[:50]])
        
        result_count = len(rows)
        if result_count > 50:
            table_rows += f"\n\n... 외 {result_count - 50}건 더 있음"
        
        response = f"**생성된 SQL:**\n```sql\n{sql}\n```\n\n**결과 ({result_count}건):**\n\n| {table_header} |\n| {table_sep} |\n| {table_rows} |"
    
    history.append((user_message, response))
    return history, ""


# Gradio UI
with gr.Blocks(
    title="Text-to-SQL 챗봇",
    theme=gr.themes.Soft(),
    css="""
    .chatbot {font-size: 14px;}
    .contain {max-width: 900px; margin: auto;}
    """
) as demo:
    gr.Markdown(
        """
        # 🗄️ MES Text-to-SQL 챗봇
        자연어로 질문하면 SQL을 생성하고 실행 결과를 보여줍니다.
        
        **예시 질문:**
        - 포르쉐라인에서 2025년 3월 오전(08~19시)과 야간(그 외) 수율을 비교해줘
        - 최근 10개 작업완료 로그 보여줘
        - 자재코드별 총 변경수량 합계
        """
    )
    
    chatbot = gr.Chatbot(
        label="대화",
        height=500,
        show_copy_button=True
    )
    
    with gr.Row():
        txt = gr.Textbox(
            label="질문 입력",
            placeholder="자연어로 질문을 입력하세요...",
            scale=9
        )
        btn = gr.Button("전송", variant="primary", scale=1)
    
    # 이벤트 연결
    txt.submit(chat, [txt, chatbot], [chatbot, txt])
    btn.click(chat, [txt, chatbot], [chatbot, txt])
    
    gr.Markdown("---\n*Model: qwen3-14b-text-to-sql-ko | DB: mesdb (MariaDB)*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)