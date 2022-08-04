import React, { Component } from 'react';
import _ from 'lodash';

export const withMouse = (WrappedComponent) => {
  return class extends Component {
    constructor(props) {
      super(props);
      this.state = {
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
      };
    }

    getMousePosition = (event) => {
      const vm = this;
      const e = event || window.event;
      vm.setState({
        top: e.clientY,
        left: e.clientX,
        right: window.innerWidth - e.clientX,
        bottom: window.innerHeight - e.clientY,
      });
    }

    handleMouseMove = _.throttle(this.getMousePosition, 300, {leading: true});

    componentDidMount() {
      window.addEventListener('mousemove', this.handleMouseMove, false);
    }

    componentWillUnmount() {
      window.removeEventListener('mousemove', this.handleMouseMove, false);
    }

    render() {
      return (
        <WrappedComponent {...this.props} position={this.state} />
      )
    }
  }
}
