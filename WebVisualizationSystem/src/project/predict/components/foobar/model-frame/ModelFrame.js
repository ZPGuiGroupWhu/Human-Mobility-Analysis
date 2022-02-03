import React, { useState, useContext } from 'react';
import "./ModelFrame.scss";
import "../css/common.css";
import { Select, Button } from 'antd';
import { MenuUnfoldOutlined, MenuFoldOutlined } from '@ant-design/icons';
import _ from 'lodash';
import { PredictCtx } from '@/project/predict/Predict';

const ModelFrame = (props) => {
  const { Option } = Select;
  const { options } = props;
  const { modalVisible, setModalVisible } = useContext(PredictCtx);

  const defaultOption = options[0];
  const [model, setModel] = useState(defaultOption.name);

  const onChange = (value) => {
    setModel(value);
  }

  const showModal = () => { setModalVisible(true) };
  const hideModal = () => { setModalVisible(false) };

  return (
    <div className="model-frame-ctn universal-frame-style">
      <div className="select-ctn">
        <Select style={{ width: 130 }} defaultValue={defaultOption.name} onChange={onChange} >
          {
            options.map(({ name }, key) => (
              <Option value={name} key={key} >{name}</Option>
            ))
          }
        </Select>
        {
          !modalVisible ? 
          (<Button icon={<MenuUnfoldOutlined />} size='small' onClick={showModal}></Button>) : 
          (<Button icon={<MenuFoldOutlined />} size='small' onClick={hideModal}></Button>)
        }
      </div>
      {
        options.find(item => item.name === model).component
      }
    </div>
  )
}

export default ModelFrame;