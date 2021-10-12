import React, { useState } from 'react';
import "./ModelFrame.scss";
import "../css/common.css";
import { Select } from 'antd';
import _ from 'lodash';

const ModelFrame = (props) => {
  const { Option } = Select;
  const { options } = props;

  const defaultOption = options[0];
  const [model, setModel] = useState(defaultOption.name);

  const onChange = (value) => {
    setModel(value);
  }

  return (
    <div className="model-frame-ctn universal-frame-style">
      <div className="select-ctn">
        <span className="select-text">模型：</span>
        <Select style={{ width: 120 }} defaultValue={defaultOption.name} onChange={onChange} >
          {
            options.map(({ name }, key) => (
              <Option value={name} key={key} >{name}</Option>
            ))
          }
        </Select>
      </div>
      {
        options.find(item => item.name === model ).component
      }
    </div>
  )
}

export default ModelFrame;