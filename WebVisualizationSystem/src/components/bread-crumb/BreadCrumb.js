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
    <span>{props.sperator}</span>
    <span style={{ position: 'relative', top: '1px' }}>{props.breadCrumbName}</span>
  </ NavLink>
)

const SpanItem = (props) => (
  <span className="item" style={props.style}>
    <span>{props.sperator}</span>
    <span style={{ position: 'relative', top: '1px' }}>{props.breadCrumbName}</span>
  </span>
)

export default function BreadCrumb(props) {
  const sperator = '> '; // 分隔符
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
    return (
      status ?
        (
          <NavLinkItem
            availableStyle={availableStyle}
            activeStyle={activeStyle}
            sperator={sperator}
            breadCrumbName={breadCrumbName}
            targetURL={targetURL}
          />
        ) :
        (
          <SpanItem
            style={forbiddenStyle}
            breadCrumbName={breadCrumbName}
            sperator={sperator}
          />
        )
    )
  }
  return (
    <span className="bread-crumb-ctn">
      {props.routes.map(item => createItem(item))}
    </span>
  )
}
