import type { ComfyNodeOutput, ComfyWorkflow } from '@/types/comfy'

export interface ApiWorkflowNode {
  inputs: Record<string, any>
  class_type: string
  _meta: {
    title: string
  }
}

export interface ApiWorkflow {
  [key: string]: ApiWorkflowNode
}

export class WorkflowConverter {
  /**
   * Converts a standard ComfyUI workflow to API format
   */
  static toApiFormat(workflow: ComfyWorkflow): ApiWorkflow {
    const apiWorkflow: ApiWorkflow = {}
    
    // Convert each node to API format
    for (const [nodeId, node] of Object.entries(workflow.nodes)) {
      const apiNode: ApiWorkflowNode = {
        inputs: {},
        class_type: node.type,
        _meta: {
          title: node.title || node.type
        }
      }

      // Convert inputs to API format
      for (const [inputName, input] of Object.entries(node.inputs || {})) {
        if (Array.isArray(input) && input.length === 2) {
          // Handle node connections [nodeId, outputIndex]
          apiNode.inputs[inputName] = input
        } else {
          // Handle direct values
          apiNode.inputs[inputName] = input
        }
      }

      apiWorkflow[nodeId] = apiNode
    }

    return apiWorkflow
  }

  /**
   * Converts an API format workflow back to standard ComfyUI format
   */
  static fromApiFormat(apiWorkflow: ApiWorkflow): ComfyWorkflow {
    const workflow: ComfyWorkflow = {
      nodes: {},
      edges: [],
      groups: [],
      config: {},
      version: 0.4
    }

    // Convert each API node to standard format
    for (const [nodeId, apiNode] of Object.entries(apiWorkflow)) {
      const node = {
        id: parseInt(nodeId),
        type: apiNode.class_type,
        title: apiNode._meta?.title || apiNode.class_type,
        pos: [0, 0], // Default position
        size: { width: 0, height: 0 }, // Default size
        inputs: {},
        outputs: {},
        properties: {}
      }

      // Convert inputs from API format
      for (const [inputName, input] of Object.entries(apiNode.inputs)) {
        if (Array.isArray(input) && input.length === 2) {
          // Handle node connections
          workflow.edges.push({
            sourceNode: input[0].toString(),
            sourceOutput: input[1],
            targetNode: nodeId,
            targetInput: inputName
          })
        } else {
          // Handle direct values
          node.inputs[inputName] = input
        }
      }

      workflow.nodes[nodeId] = node
    }

    return workflow
  }
}