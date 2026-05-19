#!/bin/bash

# MeMOTR��Ŀ�ϴ��ű�
# ���ڽ���Ŀ�ϴ���AutoDLƽ̨

echo "=== �ϴ�MeMOTR��Ŀ��AutoDL ==="

# AutoDL������Ϣ
AUTODL_HOST="connect.nmb2.seetacloud.com"
AUTODL_PORT=21178
AUTODL_USER="root"

# ������Ŀ·�������޸�Ϊ����ʵ��·����
LOCAL_PROJECT="/path/to/your/MeMOTR-main"

# Զ����Ŀ·��
REMOTE_PROJECT="/root/MeMOTR-main"

echo "������Ϣ:"
echo "����: $AUTODL_HOST"
echo "�˿�: $AUTODL_PORT"
echo "�û�: $AUTODL_U SER"
echo "������Ŀ: $LOCAL_PROJECT"
echo "Զ����Ŀ: $REMOTE_PROJECT"

echo ""
echo "��ȷ��������Ŀ·����ȷ��Ȼ��Enter����..."
read

# ʹ��rsyncͬ����Ŀ�ļ�
echo "��ʼͬ����Ŀ�ļ�..."
rsync -avz -e "ssh -p $AUTODL_PORT" \
    --exclude='.git' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    $LOCAL_PROJECT/ $AUTODL_USER@$AUTODL_HOST:$REMOTE_PROJECT/

echo "�ļ�ͬ����ɣ�"

echo ""
echo "��һ������:"
echo "1. ���ӵ�AutoDL������: ssh -p $AUTODL_PORT $AUTODL_USER@$AUTODL_HOST"
echo "2. ������ĿĿ¼: cd $REMOTE_PROJECT"
echo "3. ׼������: bash prepare_data.sh"
echo "4. ����ѵ��: bash autodl_start.sh"
echo "5. ���ѵ��: python train_monitor.py"
