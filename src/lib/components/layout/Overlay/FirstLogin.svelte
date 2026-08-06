<script lang="ts">
  export let onClose: () => void;
	import { goto } from '$app/navigation';
	import { WEBUI_API_BASE_URL } from '$lib/constants';

	const markFirstLoginDone = async () => {
		const res = await fetch(`${WEBUI_API_BASE_URL}/users/first-login/done`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				authorization: `Bearer ${localStorage.token}`
			}
		});

		if (!res.ok) {
			throw new Error('mark first login done failed');
		}

		return await res.json();
	};

	const handleCreate = async () => {
		try {
      const result = await markFirstLoginDone();
      console.log('markFirstLoginDone result:', result);
      onClose?.();
      await goto('/workspace/knowledge');
    } catch (error) {
      console.error(error);
    }
	};

	const handleLater = async () => {
		try {
			const result = await markFirstLoginDone();
			console.log('markFirstLoginDone result:', result);
      onClose?.();
			await goto('/');
		} catch (error) {
			console.error(error);
		}
	};
</script>

<div class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40">
	<div class="w-full max-w-md rounded-2xl bg-white dark:bg-gray-900 p-8 shadow-xl">
		<div class="text-center text-2xl font-medium dark:text-white">
			创建属于您的知识库
		</div>

		<div class="mt-4 text-center text-sm text-gray-600 dark:text-gray-300">
			这是您首次进入系统。<br />
			是否现在创建属于您自己的知识库？
		</div>

		<div class="mt-6 flex flex-col items-center gap-3">
			<button
				class="px-5 py-2 rounded-full bg-white border border-gray-200 hover:bg-gray-100 text-gray-700 font-medium text-sm"
				on:click={handleCreate}
			>
				去创建知识库
			</button>

			<button class="text-xs text-gray-400 underline" on:click={handleLater}>
				稍后
			</button>
		</div>
	</div>
</div>