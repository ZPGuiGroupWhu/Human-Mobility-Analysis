import React, { useReducer, useState } from 'react';
import { Card, Avatar, Drawer, Tooltip, Select } from 'antd';
import { EllipsisOutlined, SettingOutlined, PlayCircleOutlined, PauseCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import './ModelCard.scss';

export default function ModelCard(props) {
  // props
  const {
    // cover options
    width = '100%',
    imgUrl = 'https://wallhaven.cc/w/8ogod1',
    // meta options
    avatarUrl = 'https://avatars.githubusercontent.com/u/42670632?v=4',
    title = 'Model Name',
    description = 'Model Description',
    drawerHeight = '350px',
    startPredict,
    stopPredict,
    clearPredict,
  } = props

  const keys = ['参数配置', '模型选择', '开始预测', '暂停预测', '清除']

  // components
  const { Meta } = Card;

  // hooks
  const [optionsVisible, setOptionsVisible] = useState(false);
  const [settingVisible, setSettingVisible] = useState(false);
  const [tooltip, setTooltips] = useState('');
  // MODEL SETTING
  function modelReducer(state, action) {
    switch (action.type) {
      case 'model1':
        return (
          <div>model1</div>
        )
      case 'model2':
        return (
          <div>model2</div>
        )
      case 'model3':
        return (
          <div>model3</div>
        )
      case 'model4':
        return (
          <div>model4</div>
        )
      default:
        return null;
    }
  }
  const [modelComp, modelDispatch] = useReducer(modelReducer, null)
  // 模型是否已经选择
  const [modelSelected, setModelSelected] = useState(false);

  // function



  
  return (
    <div
      style={{ width }}
      className='model-card-container'
    >
      <Card
        className='card-style'
        cover={
          <img
            alt=""
            src={imgUrl}
          />
        }
        actions={[
          <a
            onClick={(e) => {
              e.preventDefault(); modelSelected && setSettingVisible(true)
            }}
            onMouseOver={(e) => { setTooltips(keys[0]) }}
            onMouseLeave={() => { setTooltips('') }}
            style={{ cursor: modelSelected ? 'pointer' : 'not-allowed' }}
          >
            <Tooltip placement='top' title={keys[0]} visible={tooltip === keys[0]}>
              <SettingOutlined key={keys[0]} />
            </Tooltip>
          </a>,
          <a
            onClick={(e) => {
              e.preventDefault(); setOptionsVisible(true);
            }}
            onMouseOver={() => setTooltips(keys[1])}
            onMouseLeave={() => setTooltips('')}
          >
            <Tooltip placement='top' title={keys[1]} visible={tooltip === keys[1]}>
              <EllipsisOutlined key={keys[1]} />
            </Tooltip>
          </a>,
          <a
            onClick={(e) => { 
              e.preventDefault();
              modelSelected && startPredict();
            }}
            onMouseOver={() => setTooltips(keys[2])}
            onMouseLeave={() => setTooltips('')}
            style={{ cursor: modelSelected ? 'pointer' : 'not-allowed' }}
          >
            <Tooltip placement='top' title={keys[2]} visible={tooltip === keys[2]}>
              <PlayCircleOutlined key={keys[2]} />
            </Tooltip>
          </a>,
          <a
            onClick={(e) => { 
              e.preventDefault(); 
              modelSelected && stopPredict();
            }}
            onMouseOver={() => setTooltips(keys[3])}
            onMouseLeave={() => setTooltips('')}
            style={{ cursor: modelSelected ? 'pointer' : 'not-allowed' }}
          >
            <Tooltip placement='top' title={keys[3]} visible={tooltip === keys[3]}>
              <PauseCircleOutlined key={keys[3]} />
            </Tooltip>
          </a>,
          <a
            onClick={(e) => { 
              e.preventDefault();
              modelSelected && clearPredict(); 
            }}
            onMouseOver={() => setTooltips(keys[4])}
            onMouseLeave={() => setTooltips('')}
            style={{ cursor: modelSelected ? 'pointer' : 'not-allowed' }}
          >
            <Tooltip placement='top' title={keys[4]} visible={tooltip === keys[4]}>
              <CloseCircleOutlined key={keys[4]} />
            </Tooltip>
          </a>,
        ]}
      >
        <Meta
          avatar={<Avatar src={avatarUrl} />}
          title={title}
          description={description}
        />
      </Card>
      {/* 模型选项弹窗 */}
      <Drawer
        title="模型选项"
        placement="top"
        closable={true}
        onClose={() => setOptionsVisible(false)}
        visible={optionsVisible}
        getContainer={false}
        style={{ position: 'absolute' }}
        width='100%'
        height={drawerHeight}
      >
        <Select
          style={{ width: '100%' }}
          allowClear
          onSelect={val => {
            modelDispatch({ type: val });
            setModelSelected(true);
            setTimeout(() => {
              setOptionsVisible(false);
            }, 200)
          }}
          onClear={() => setModelSelected(false)}
        >
          <Select.Option value="model1">Model 1</Select.Option>
          <Select.Option value="model2">Model 2</Select.Option>
          <Select.Option value="model3">Model 3</Select.Option>
          <Select.Option value="model4">Model 4</Select.Option>
        </Select>
      </Drawer>
      {/* 模型设置弹窗 */}
      <Drawer
        title="模型参数设置"
        placement="top"
        closable={true}
        onClose={() => setSettingVisible(false)}
        visible={settingVisible}
        getContainer={false}
        style={{ position: 'absolute' }}
        width='100%'
        height={drawerHeight}
      >
        {modelComp}
      </Drawer>
    </div>
  )
}
