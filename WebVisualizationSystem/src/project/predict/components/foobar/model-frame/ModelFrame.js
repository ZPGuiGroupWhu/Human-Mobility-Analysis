import React from 'react';
import "./ModelFrame.scss";
import "../css/common.css";
import { Select } from 'antd';
import _ from 'lodash';

const ModelFrame = (props) => {
  const { Option } = Select;
  const { options } = props;

  return (
    <div className="model-frame-ctn universal-frame-style">
      <div className="select-ctn">
        <span className="select-text">当前模型：</span>
        <Select defaultValue="" style={{ width: 80 }} >
          {
            options.map((item, key) => (<Option value={item} key={key} >{item}</Option>))
          }
        </Select>
      </div>
    </div>
  )
}

export default ModelFrame;