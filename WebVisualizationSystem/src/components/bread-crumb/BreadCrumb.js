import React from 'react';
import { NavLink } from 'react-router-dom';
import './BreadCrumb.scss';

const NavLinkItem = (props) => (
  <NavLink
    className="item"
    style={props.availableStyle}
    activeStyle={props.activeStyle}
    to={location => {
      return {
        ...location,
        pathname: props.targetURL,
      }
    }
    }
    exact
  >
    {props.sperator + props.breadCrumbName}
  </ NavLink>
)

const SpanItem = (props) => (
  <span className="item" style={props.style}>{props.sperator + props.breadCrumbName}</span>
)

export default function BreadCrumb(props) {
  const sperator = '/ '; // 分隔符
  const activeStyle = {
    color: '#15FBF1',
    fontWeight: 'bold',
  };
  const forbiddenStyle = {
    color: '#D8DADA',
    cursor: 'not-allowed',
  };
  const availableStyle = {
    color: '#fff',
  }

  const createItem = ({ breadCrumbName, targetURL, status }) => {
    const { available, forbidden } = status;
    const Item = () => {
      switch (true) {
        case (available):
          return (
            <NavLinkItem
              availableStyle={availableStyle}
              activeStyle={activeStyle}
              sperator={sperator}
              breadCrumbName={breadCrumbName}
              targetURL={targetURL}
            />
          )
        case forbidden:
          return (
            <SpanItem
              style={available ? availableStyle : (forbidden ? forbiddenStyle : '')}
              breadCrumbName={breadCrumbName}
              sperator={sperator}
            />
          )
        default:
          return;
      }
    }
    return <Item />
  }
  return (
    <span className="bread-crumb-ctn">
      {props.routes.map(item => createItem(item))}
    </span>
  )
}
