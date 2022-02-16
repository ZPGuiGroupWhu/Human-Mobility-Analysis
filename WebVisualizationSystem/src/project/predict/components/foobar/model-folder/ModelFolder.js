import React, { useState, useContext, useEffect } from 'react';
import { Select, Button, Tooltip } from 'antd';
import { MenuUnfoldOutlined, MenuFoldOutlined } from '@ant-design/icons';
import _ from 'lodash';
import './ModelFolder.scss';
import '../common.css';
import { PredictCtx } from '@/project/predict/Predict';
import FoldPanel from '@/components/fold-panel/FoldPanel';
import PredictFrame from '@/project/predict/components/foobar/predict-frame/PredictFrame';
import Map3D from '@/project/predict/components/deckgl/Map3D.js'; // DeckGL

const ModelFrame = (props) => {
  const { Option } = Select;
  const { options, onPredictDispatch } = props;
  const { modalVisible, setModalVisible } = useContext(PredictCtx);

  const defaultOption = options[0];
  const [model, setModel] = useState(defaultOption.name);

  const onChange = (value, setFold) => {
    setModel(value);
    setFold(true);
    setTimeout(() => {
      setFold(false);
    }, 0);
  }

  const showModal = () => { setModalVisible(true) };
  const hideModal = () => { setModalVisible(false) };

  return (
    <FoldPanel
      width='100%'
      id='model-folder'
      className='common-margin-bottom'
      renderEntryComponent={(setFold) => (
        <div id='model-folder-entry'>
          <div id="model-folder-model-selector">
            <Select style={{ width: 130 }} onChange={(val) => onChange(val, setFold)} >
              {
                options.map(({ name }, key) => (
                  <Option value={name} key={key} >{name}</Option>
                ))
              }
            </Select>
            {
              !modalVisible ?
                (<Tooltip title='模型介绍'><Button icon={<MenuUnfoldOutlined />} size='small' onClick={showModal}></Button></Tooltip>) :
                (<Button icon={<MenuFoldOutlined />} size='small' onClick={hideModal}></Button>)
            }

          </div>
          <PredictFrame onPredictDispatch={onPredictDispatch} />
          {/* 轨迹3d展示 */}
          <Map3D />
        </div>
      )}
      renderExpandComponent={() => (
        <div id='model-folder-expand'>
          {options.find(item => item.name === model).component}
        </div>
      )}
    />
  )
}

export default ModelFrame;